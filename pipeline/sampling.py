import pandas as pd
from imblearn.over_sampling import SMOTE
import logging
from typing import Tuple

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) oversampling
    to the feature matrix X and label vector y.

    Args:
        X (pd.DataFrame): The input feature matrix.
        y (pd.Series): The target label vector.
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: A tuple containing the resampled
                                         feature matrix (X_res) and label vector (y_res).

    Raises:
        ValueError: If input data is not suitable for SMOTE (e.g., all one class).
        Exception: For other unexpected errors during SMOTE application.
    """
    if X.empty or y.empty:
        logger.warning("Input data for SMOTE is empty. Returning original empty data.")
        return X, y

    if y.nunique() < 2:
        logger.warning(f"SMOTE cannot be applied: only one class ({y.iloc[0]}) found in target variable. Returning original data.")
        return X, y

    # Check if any class has too few samples for SMOTE (e.g., less than k_neighbors, default 5)
    # SMOTE default k_neighbors is 5. If any class has < 6 samples, it will raise an error.
    # It's better to log a warning than to crash.
    min_samples_per_class = y.value_counts().min()
    if min_samples_per_class < 6: # Default k_neighbors + 1
        logger.warning(f"SMOTE might fail: Smallest class has only {min_samples_per_class} samples. Consider adjusting k_neighbors or skipping SMOTE.")

    try:
        sm = SMOTE(random_state=random_state)
        X_res, y_res = sm.fit_resample(X, y)
        logger.info(f"[SAMPLE] SMOTE applied: Resampled from {len(X)} to {len(X_res)} samples. Class distribution (original vs resampled): {y.value_counts().to_dict()} vs {y_res.value_counts().to_dict()}")
        return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name) # Ensure DataFrames/Series are returned
    except ValueError as e:
        logger.error(f"ValueError during SMOTE application: {e}. Check class distribution or number of samples.")
        raise ValueError(f"SMOTE application failed due to data characteristics: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during SMOTE application: {e}")
        raise RuntimeError(f"SMOTE failed: {e}")


def downsample_df(df: pd.DataFrame, max_samples: int, random_state: int = 42) -> pd.DataFrame:
    """
    Randomly downsample a DataFrame to `max_samples` rows if its current size
    exceeds that limit. This maintains global class proportion by sampling
    without replacement, assuming 'Label' column exists.

    Args:
        df (pd.DataFrame): The input DataFrame to potentially downsample.
        max_samples (int): The maximum number of samples allowed.
        random_state (int): Seed for random number generation for reproducibility.

    Returns:
        pd.DataFrame: The original DataFrame if its size is within `max_samples`,
                      or a downsampled DataFrame.

    Raises:
        ValueError: If `max_samples` is non-positive or if 'Label' column is missing for stratified sampling.
        Exception: For other unexpected errors during downsampling.
    """
    if not isinstance(max_samples, int) or max_samples <= 0:
        logger.error(f"Invalid `max_samples` value: {max_samples}. Must be a positive integer.")
        raise ValueError("`max_samples` must be a positive integer.")

    if df.empty:
        logger.warning("Input DataFrame for downsampling is empty. Returning empty DataFrame.")
        return df

    if len(df) <= max_samples:
        logger.debug(f"[SAMPLE] DataFrame size ({len(df)}) is within or equal to max_samples ({max_samples}). No downsampling applied.")
        return df

    try:
        # Check for 'Label' column for stratified sampling
        if "Label" in df.columns and df["Label"].nunique() > 1:
            # Stratified sampling if 'Label' exists and has multiple classes
            # Ensure each class has enough samples for stratified sampling
            min_class_count = df["Label"].value_counts().min()
            if min_class_count < 2: # Need at least 2 samples per class for robust stratified sampling
                logger.warning(f"One or more classes in 'Label' has fewer than 2 samples ({min_class_count}). Falling back to simple random sampling.")
                df_downsampled = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
            else:
                # Calculate sample proportion for each class to maintain distribution
                fractions = df["Label"].value_counts(normalize=True) * max_samples
                fractions = fractions.apply(lambda x: int(round(x))) # Round to nearest integer samples per class

                # Adjust total samples if rounding leads to discrepancy
                total_sampled = fractions.sum()
                if total_sampled > max_samples: # Trim if too many
                    diff = total_sampled - max_samples
                    # Reduce from largest classes first
                    sorted_classes = fractions.sort_values(ascending=False).index
                    for cls in sorted_classes:
                        if diff == 0: break
                        if fractions[cls] > 0:
                            fractions[cls] -= 1
                            diff -= 1
                elif total_sampled < max_samples: # Add if too few
                    diff = max_samples - total_sampled
                    # Add to largest classes first
                    sorted_classes = fractions.sort_values(ascending=False).index
                    for cls in sorted_classes:
                        if diff == 0: break
                        fractions[cls] += 1
                        diff -= 1

                sampled_parts = []
                for label_val, n_samples in fractions.items():
                    if n_samples > 0:
                        class_df = df[df["Label"] == label_val]
                        if n_samples > len(class_df):
                            logger.warning(f"Requested {n_samples} samples for class {label_val}, but only {len(class_df)} available. Sampling all available.")
                            n_samples = len(class_df)
                        sampled_parts.append(class_df.sample(n=n_samples, random_state=random_state))

                df_downsampled = pd.concat(sampled_parts).sample(frac=1, random_state=random_state).reset_index(drop=True) # Shuffle after concat

        else:
            if "Label" not in df.columns:
                logger.warning("No 'Label' column found for stratified downsampling. Performing simple random sampling.")
            elif df["Label"].nunique() <= 1:
                logger.info(f"Only one class in 'Label' column ({df['Label'].nunique()} unique values). Performing simple random sampling.")
            df_downsampled = df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)

        logger.info(f"[SAMPLE] Downsampled from {len(df)} to {len(df_downsampled)} rows.")
        return df_downsampled
    except Exception as e:
        logger.error(f"An unexpected error occurred during downsampling: {e}")
        raise RuntimeError(f"Downsampling failed: {e}")

# Example Usage (for demonstration)
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    print("\n--- Demo: apply_smote ---")
    # Create a highly imbalanced dataset
    data_imbalanced = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'Label': [0]*95 + [1]*5
    }
    df_imbalanced = pd.DataFrame(data_imbalanced)
    X_imbalanced = df_imbalanced[['feature1', 'feature2']]
    y_imbalanced = df_imbalanced['Label']

    logger.info(f"Original class distribution: {y_imbalanced.value_counts().to_dict()}")
    try:
        X_resampled, y_resampled = apply_smote(X_imbalanced, y_imbalanced)
        logger.info(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
        logger.info(f"New dataset size after SMOTE: {len(X_resampled)} samples.")
    except Exception as e:
        logger.error(f"SMOTE demo failed: {e}")

    print("\n--- Demo: downsample_df ---")
    # Create a large balanced dataset
    data_large = {
        'feature1': np.random.rand(20000),
        'feature2': np.random.rand(20000),
        'Label': np.random.randint(0, 2, 20000)
    }
    df_large = pd.DataFrame(data_large)

    max_samples_target = 5000
    logger.info(f"Original DataFrame size: {len(df_large)}")
    logger.info(f"Original class distribution: {df_large['Label'].value_counts().to_dict()}")

    try:
        df_downsampled = downsample_df(df_large, max_samples_target)
        logger.info(f"Downsampled DataFrame size: {len(df_downsampled)}")
        logger.info(f"Downsampled class distribution: {df_downsampled['Label'].value_counts().to_dict()}")
    except Exception as e:
        logger.error(f"Downsample demo failed: {e}")

    print("\n--- Demo: downsample_df (no downsample needed) ---")
    df_small = pd.DataFrame({'feature': [1,2,3], 'Label': [0,1,0]})
    df_no_downsample = downsample_df(df_small, 100)
    logger.info(f"No downsample needed. Resulting size: {len(df_no_downsample)}")

    print("\n--- Demo: downsample_df (missing Label) ---")
    df_no_label = pd.DataFrame({'feature': np.random.rand(100), 'val': np.random.rand(100)})
    try:
        df_no_label_downsampled = downsample_df(df_no_label, 50)
        logger.info(f"Downsampled (no label): {len(df_no_label_downsampled)}")
    except Exception as e:
        logger.error(f"Downsample (no label) demo failed: {e}")

