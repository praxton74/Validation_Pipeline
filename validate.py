# ============================================
#   Outlier and Skewness Validation Pipeline
# ============================================
# PEP8 Compliant, Readable, Structured, and Well-Commented

# -------- Step 1: Import Required Libraries --------
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from datetime import datetime

#   Prepare today's date string for consistent output filenames
TODAY_STR = datetime.now().strftime("%d%m%Y")

# -------- Step 2: Utility Functions --------

def validate_features_exist(df, features):
    """Validates that the specified features exist in the DataFrame.

    Returns cleaned feature names and missing features (if any).
    """
    df.columns = df.columns.str.strip().str.lower()
    cleaned = [f.strip().lower() for f in features]
    missing = [f for f in cleaned if f not in df.columns]
    return cleaned, missing

def get_safe_feature_ranges(df, features, method="buffer", buffer=0.5,
                             lower_pct=0.01, upper_pct=0.99):
    """Calculates acceptable value ranges using buffer or percentiles."""
    ranges = {}
    cleaned_features, _ = validate_features_exist(df, features)
    for feature in cleaned_features:
        if method == "buffer":
            min_val = df[feature].min() - buffer
            max_val = df[feature].max() + buffer
        elif method == "percentile":
            min_val = df[feature].quantile(lower_pct)
            max_val = df[feature].quantile(upper_pct)
        else:
            raise ValueError("Unknown method")
        ranges[feature] = (round(min_val, 4), round(max_val, 4))
    return ranges

def validate_feature_range(df, feature, expected_min, expected_max,
                            log_outliers=True, visualize=False,
                            log_filename=None, global_outliers=None):
    """Validates if a feature lies within the expected range.

    Logs outliers and optionally visualizes them with a boxplot.
    """
    results = {
        'feature': feature,
        'total': len(df),
        'outlier_count': 0,
        'outlier_ratio': 0.0,
        'status': 'Unchecked'
    }

    if feature not in df.columns:
        results['status'] = 'Missing'
        return results

    mask = (df[feature] < expected_min) | (df[feature] > expected_max)
    outliers = df[mask]
    count = len(outliers)

    results.update({
        'outlier_count': count,
        'outlier_ratio': round(count / len(df), 4),
        'status': 'OK' if count == 0 else 'Outliers found'
    })

    if count > 0 and log_outliers and log_filename:
        outliers.to_csv(log_filename, index=False)
        sparse_outliers = df.loc[mask].copy()
        for col in sparse_outliers.columns:
            if col != feature:
                sparse_outliers[col] = np.nan
        sparse_filename = log_filename.replace(".csv", "_sparse.csv")
        sparse_outliers.to_csv(sparse_filename, index=False)

        if global_outliers is not None:
            for idx, row in outliers.iterrows():
                global_outliers.append({
                    'id': row.get('id', idx),
                    'feature': feature,
                    'value': row[feature],
                    'expected_min': expected_min,
                    'expected_max': expected_max,
                    'date': datetime.now().strftime("%Y-%m-%d")
                })

    if visualize:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=df[feature], ax=ax)
        ax.set_title(f"Boxplot — {feature}")
        plt.tight_layout()
        plt.show()

    return results

def batch_validate_features(df, feature_ranges, log_dir=None, visualize=False):
    """Validates multiple features and logs outliers."""
    if log_dir is None:
        log_dir = f"{TODAY_STR}_outlier_logs"
    os.makedirs(log_dir, exist_ok=True)

    results = []
    global_outliers = []

    for feature, (min_val, max_val) in feature_ranges.items():
        log_file = os.path.join(log_dir, f"{feature}_outliers.csv")
        result = validate_feature_range(
            df, feature, min_val, max_val,
            log_outliers=True,
            visualize=visualize,
            log_filename=log_file,
            global_outliers=global_outliers
        )
        results.append(result)

    pd.DataFrame(global_outliers).to_csv(
        f"{TODAY_STR}_all_outliers_combined.csv", index=False
    )
    return pd.DataFrame(results)

# -------- Step 3: Combined Pipeline --------

def combined_outlier_skew_pipeline(df, skew_thresh=1.0):
    """Full pipeline:
    1. Detect skewed numeric features
    2. Apply log1p transformation
    3. Define safe ranges
    4. Validate and log outliers
    """
    print("\n  Step 1: Detecting skewed features...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    skew_vals = df[numeric_cols].apply(lambda x: skew(x.dropna()))
    skewed_feats = skew_vals[skew_vals > skew_thresh].index
    print(f"  Found {len(skewed_feats)} highly skewed features: {list(skewed_feats)}")

    print("\n  Step 2: Applying log1p transformation to skewed features...")
    df_transformed = df.copy()
    df_transformed[skewed_feats] = df_transformed[skewed_feats].apply(np.log1p)

    print("\n  Step 3: Generating safe ranges...")
    ranges = get_safe_feature_ranges(
        df_transformed,
        skewed_feats,
        method="percentile",
        lower_pct=0.01,
        upper_pct=0.99
    )

    print("\n  Step 4: Validating and logging outliers...")
    report = batch_validate_features(
        df_transformed,
        feature_ranges=ranges,
        visualize=True
    )

    report.to_csv(
        f"{TODAY_STR}_auto_validation_summary.csv", index=False
    )
    print(f"\n  Validation summary saved as '{TODAY_STR}_auto_validation_summary.csv'")

    return df_transformed, report

# -------- Step 4: Main Driver Block --------

if __name__ == "__main__":
    df = pd.read_csv("Wisconsin_Breast_Cancer_Dataset.csv")
    df_cleaned, validation_report = combined_outlier_skew_pipeline(df)
    print(validation_report.head())
