import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
import shap

# --- Configuration ---
RESULTS_DIR = Path("results")
CLEANED_DIR = Path("cleaned")
PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(exist_ok=True)

# Sampling cap per dataset for meta-model computations
SAMPLE_PER_DATASET = 100_000

# Load CSV results once
print("[INIT] Loading result CSVs...")
base_overall = pd.read_csv(RESULTS_DIR / "base_models" / "overall_metrics.csv")
base_per_attack = pd.read_csv(RESULTS_DIR / "base_models" / "per_attack_metrics.csv")
meta_within = pd.read_csv(RESULTS_DIR / "meta_stack" / "within_day_results.csv")
meta_cross = pd.read_csv(RESULTS_DIR / "meta_stack" / "cross_day_results.csv")
print("[INIT] CSVs loaded.")

# --- 1. Cross-Day Trend Plot ---
def plot_cross_day_trend():
    print("[1] Starting Cross-Day Trend Plot...")
    df = meta_cross.copy()
    df["sampling"] = df["sampling"].fillna("None")
    df["include_meta"] = df["include_meta"].astype(str)
    df["experiment"] = (
        df["meta_model"] +
        " | Meta: " + df["include_meta"] +
        " | Sampling: " + df["sampling"]
    )
    print("[1] Data prepared, creating lineplot...")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="leave_out", y="f1", hue="experiment", marker="o")
    plt.title("Cross-Day F1 Scores by Meta-Model Configuration")
    plt.xlabel("Test Day (Leave-One-Out)")
    plt.ylabel("F1 Score")
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    path = PLOTS_DIR / "cross_day_f1_trend.png"
    plt.savefig(path)
    plt.close()
    print(f"[1] Cross-Day Trend saved to {path}.")

# --- 2. Feature Correlation Heatmap ---
def plot_feature_correlation():
    print("[2] Starting Feature Correlation Heatmap...")
    csv_files = list(CLEANED_DIR.glob("*.csv"))
    all_corrs = []
    for file in csv_files:
        print(f"[2] Computing correlation for {file.name}...")
        try:
            df = pd.read_csv(file)
            numeric = df.select_dtypes(include=np.number).drop(columns=["Label"], errors="ignore")
            corr = numeric.corr().fillna(0)
            all_corrs.append(corr)
        except Exception as e:
            print(f"[2] Skipping {file.name}: {e}")
    if not all_corrs:
        print("[2] No correlations to average, exiting.")
        return
    avg_corr = sum(all_corrs) / len(all_corrs)
    print("[2] Plotting average correlation heatmap...")
    plt.figure(figsize=(14, 12))
    sns.heatmap(avg_corr, cmap="coolwarm", center=0)
    plt.title("Average Feature Correlation Heatmap (All Days)")
    plt.tight_layout()
    path = PLOTS_DIR / "feature_correlation_heatmap_avg.png"
    plt.savefig(path)
    plt.close()
    print(f"[2] Feature Correlation saved to {path}.")

# --- 3. PCA and t-SNE Visualizations ---
def plot_pca_tsne():
    print("[3] Starting PCA and t-SNE projection...")
    all_data = []
    for file in CLEANED_DIR.glob("*.csv"):
        print(f"[3] Loading {file.name} for dimensionality reduction...")
        try:
            df = pd.read_csv(file)
            df = df.select_dtypes(include=np.number).dropna()
            if "Label" not in df.columns:
                continue
            all_data.append(df)
        except Exception as e:
            print(f"[3] Skipping {file.name}: {e}")
    if not all_data:
        print("[3] No data available for PCA/t-SNE, exiting.")
        return
    df_full = pd.concat(all_data, ignore_index=True)
    max_total = 50_000
    print(f"[3] Sampling up to {max_total} total rows while preserving class imbalance...")
    sampled_df = (
        df_full.groupby("Label", group_keys=False)
               .apply(lambda x: x.sample(frac=min(1.0, max_total/len(df_full)), random_state=42))
    )
    X = sampled_df.drop(columns=["Label"]);
    y = sampled_df["Label"]
    scaler = StandardScaler(); X_scaled = scaler.fit_transform(X)
    # PCA
    print("[3] Performing PCA...")
    pca = PCA(n_components=2); pca_result = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=pca_result[:,0], y=pca_result[:,1], hue=y, alpha=0.4, s=10)
    plt.title("PCA Projection (Benign vs Attack)")
    plt.tight_layout()
    pca_path = PLOTS_DIR / "pca_projection.png"
    plt.savefig(pca_path); plt.close(); print(f"[3] PCA saved to {pca_path}.")
    # t-SNE
    print("[3] Performing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(X_scaled)
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=tsne_result[:,0], y=tsne_result[:,1], hue=y, alpha=0.4, s=10)
    plt.title("t-SNE Projection (Benign vs Attack)")
    plt.tight_layout()
    tsne_path = PLOTS_DIR / "tsne_projection.png"
    plt.savefig(tsne_path); plt.close(); print(f"[3] t-SNE saved to {tsne_path}.")

# --- 4. SHAP Summary Plot for Meta Model ---
def plot_shap_summary():
    from run_full_stack_pipeline import ALL_FILES, BASE_MODELS, META_MODELS, load_and_clean
    print("[4] Starting SHAP summary generation...")
    X_meta, y_meta = [], []
    for file in ALL_FILES:
        print(f"[4] Loading and sampling {file.name}...")
        df = load_and_clean(file)
        df = df.dropna(subset=["Label"])
        if len(df) > SAMPLE_PER_DATASET:
            df = df.sample(n=SAMPLE_PER_DATASET, random_state=42).reset_index(drop=True)
        base_preds = {}
        for model_key, (model_fn, feature_list) in BASE_MODELS.items():
            df_sub = df.dropna(subset=feature_list)
            if df_sub.empty:
                print(f"[4] {model_key}: no data after dropna, skipping.")
                continue
            model = model_fn()
            model.fit(df_sub[feature_list], df_sub["Label"])
            base_preds[model_key] = model.predict_proba(df_sub[feature_list])[:,1]
        if len(base_preds) != len(BASE_MODELS):
            print(f"[4] Incomplete base predictions for {file.name}, skipping file.")
            continue
        preds_df = pd.DataFrame(base_preds)
        preds_df["Label"] = df_sub["Label"].values[:len(preds_df)]
        X_meta.append(preds_df.drop(columns=["Label"]))
        y_meta.append(preds_df["Label"])
    X_final = pd.concat(X_meta, ignore_index=True)
    y_final = pd.concat(y_meta, ignore_index=True)
    print("[4] Training meta-model for SHAP...")
    meta_model = META_MODELS["RF_meta"](); meta_model.fit(X_final, y_final)
    print("[4] Computing SHAP values (this can take time)...")
    explainer = shap.TreeExplainer(meta_model)
    shap_values = explainer.shap_values(X_final)
    plt.figure()
    shap.summary_plot(shap_values[1], X_final, show=False)
    plt.title("SHAP Summary Plot (True Meta Model)")
    plt.tight_layout()
    shp_path = PLOTS_DIR / "shap_summary_meta_model.png"
    plt.savefig(shp_path); plt.close(); print(f"[4] SHAP summary saved to {shp_path}.")

# --- 5. Combined Base and Meta Confusion Matrices (Cross-Day) ---
def plot_all_confusion_matrices():
    from run_full_stack_pipeline import ALL_FILES, BASE_MODELS, META_MODELS, load_and_clean
    from sklearn.metrics import confusion_matrix
    print("[5] Starting confusion matrix generation...")
    base_cm_totals = {k: [] for k in BASE_MODELS.keys()}
    meta_cms = []
    for test_file in ALL_FILES:
        print(f"[5] Leave-out test: {test_file.name}")
        train_files = [f for f in ALL_FILES if f != test_file]
        # Train base models
        base_models_trained = {}
        for model_key, (model_fn, feature_list) in BASE_MODELS.items():
            X_train, y_train = [], []
            for file in train_files:
                df = load_and_clean(file)
                df = df.dropna(subset=feature_list+['Label'])
                if len(df) > SAMPLE_PER_DATASET:
                    df = df.sample(n=SAMPLE_PER_DATASET, random_state=42).reset_index(drop=True)
                if df.empty: continue
                X_train.append(df[feature_list]); y_train.append(df['Label'])
            if not X_train: continue
            X_train_final = pd.concat(X_train, ignore_index=True)
            y_train_final = pd.concat(y_train, ignore_index=True)
            model = model_fn(); model.fit(X_train_final, y_train_final)
            base_models_trained[model_key] = model
            print(f"[5] Trained base model: {model_key}")
        # Prepare meta-data
        X_meta_train, y_meta_train = [], []
        for file in train_files:
            df = load_and_clean(file)
            df = df.dropna(subset=['Label'])
            if len(df)> SAMPLE_PER_DATASET:
                df = df.sample(n=SAMPLE_PER_DATASET, random_state=42).reset_index(drop=True)
            preds = {}
            for mk,m in base_models_trained.items():
                feat=BASE_MODELS[mk][1]; df_sub=df.dropna(subset=feat)
                if df_sub.empty: continue
                preds[mk]=m.predict_proba(df_sub[feat])[:,1]
            if len(preds)!=len(base_models_trained): continue
            pdf=pd.DataFrame(preds); pdf['Label']=df_sub['Label'].values[:len(pdf)]
            X_meta_train.append(pdf.drop(columns=['Label'])); y_meta_train.append(pdf['Label'])
        Xm = pd.concat(X_meta_train, ignore_index=True)
        ym = pd.concat(y_meta_train, ignore_index=True)
        meta_model = META_MODELS['RF_meta'](); meta_model.fit(Xm, ym)
        print(f"[5] Trained meta-model on {test_file.name} leave-out.")
        # Test on this day
        df_test = load_and_clean(test_file)
        df_test = df_test.dropna(subset=['Label'])
        if len(df_test)> SAMPLE_PER_DATASET:
            df_test=df_test.sample(n=SAMPLE_PER_DATASET,random_state=42).reset_index(drop=True)
        meta_preds = {}
        for mk, m in base_models_trained.items():
            feat=BASE_MODELS[mk][1]; df_sub=df_test.dropna(subset=feat)
            if df_sub.empty: continue
            # base cm
            cm=confusion_matrix(df_sub['Label'], m.predict(df_sub[feat]), labels=[0,1])
            base_cm_totals[mk].append(cm)
            meta_preds[mk]=m.predict_proba(df_sub[feat])[:,1]
        if len(meta_preds)!=len(base_models_trained): continue
        pdf_test=pd.DataFrame(meta_preds); pdf_test['Label']=df_sub['Label'].values[:len(pdf_test)]
        y_pred_meta=meta_model.predict(pdf_test.drop(columns=['Label']))
        cm_meta=confusion_matrix(pdf_test['Label'], y_pred_meta, labels=[0,1])
        meta_cms.append(cm_meta)
        print(f"[5] Appended confusion for {test_file.name}.")
    # Plot base averages
    for mk, cms in base_cm_totals.items():
        if not cms: continue
        avg_cm=np.mean(cms,axis=0)
        plt.figure(figsize=(6,5))
        sns.heatmap(avg_cm,annot=True,fmt='.0f',cmap='Purples',xticklabels=['Benign','Attack'],yticklabels=['Benign','Attack'])
        plt.title(f"Average Confusion Matrix ({mk})")
        plt.tight_layout(); path=PLOTS_DIR/f"base_confusion_matrix_avg_{mk}.png"; plt.savefig(path); plt.close()
        print(f"[5] Saved base heatmap for {mk} to {path}.")
    # Plot meta average
    if meta_cms:
        avg_meta=np.mean(meta_cms,axis=0)
        plt.figure(figsize=(6,5))
        sns.heatmap(avg_meta,annot=True,fmt='.0f',cmap='Blues',xticklabels=['Benign','Attack'],yticklabels=['Benign','Attack'])
        plt.title("Average Confusion Matrix (Meta Model - Cross Day)")
        plt.tight_layout(); path=PLOTS_DIR/"meta_confusion_matrix_avg.png"; plt.savefig(path); plt.close()
        print(f"[5] Saved meta heatmap to {path}.")

# --- 6. Per-Port Performance (using actual Dst Port) ---
def plot_per_port_performance():
    from run_full_stack_pipeline import ALL_FILES, BASE_MODELS, META_MODELS, load_and_clean
    print("[6] Starting per-port performance plot...")
    all_port_stats = []
    for test_file in ALL_FILES:
        print(f"[6] Evaluating ports for {test_file.name}...")
        df = load_and_clean(test_file).dropna(subset=['Label','Dst Port'])
        # no downsampling here
        base_preds={}
        for mk,(fn,fl) in BASE_MODELS.items():
            df_sub=df.dropna(subset=fl+['Label'])
            if df_sub.empty: continue
            m=fn(); m.fit(df_sub[fl],df_sub['Label'])
            base_preds[mk]=m.predict_proba(df_sub[fl])[:,1]
        if len(base_preds)!=len(BASE_MODELS): continue
        pdf=pd.DataFrame(base_preds); pdf['Label']=df['Label'].values[:len(pdf)]
        X_meta=pdf.drop(columns=['Label']); y_true=pdf['Label']
        m_meta=META_MODELS['RF_meta'](); m_meta.fit(X_meta,y_true)
        y_pred=m_meta.predict(X_meta)
        df_eval=df.iloc[:len(y_pred)].copy(); df_eval['y_true']=y_true.values; df_eval['y_pred']=y_pred
        df_eval['port_bin']=pd.cut(df_eval['Dst Port'],bins=[0,1023,49151,65535],labels=['Well-known','Registered','Dynamic'])
        stats=df_eval.groupby('port_bin').apply(lambda g:pd.Series({
            'TP':np.sum((g['y_true']==1)&(g['y_pred']==1)),
            'FP':np.sum((g['y_true']==0)&(g['y_pred']==1)),
            'FN':np.sum((g['y_true']==1)&(g['y_pred']==0)),
            'TN':np.sum((g['y_true']==0)&(g['y_pred']==0))
        }))
        stats['F1']=2*stats['TP']/(2*stats['TP']+stats['FP']+stats['FN']+1e-10)
        stats.reset_index(inplace=True); all_port_stats.append(stats)
    combined=pd.concat(all_port_stats)
    avg_stats=combined.groupby('port_bin')['F1'].mean().reset_index()
    print("[6] Plotting per-port F1 scores...")
    plt.figure(figsize=(8,5))
    sns.barplot(data=avg_stats,x='port_bin',y='F1',palette='viridis')
    plt.title("Average F1 Score by Destination Port Category (Meta Model)")
    plt.tight_layout(); path=PLOTS_DIR/"per_port_performance_meta.png"; plt.savefig(path); plt.close()
    print(f"[6] Saved per-port performance to {path}.")

# --- Main Execution ---
if __name__ == "__main__":
    # plot_cross_day_trend()
    # plot_feature_correlation()
    plot_pca_tsne()
    # plot_shap_summary()
    # plot_all_confusion_matrices()
    # plot_per_port_performance()
    print("[DONE] All plots generated and saved to:", PLOTS_DIR.resolve())
