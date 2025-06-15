Outlier and Skewness Validation Pipeline

This project provides a robust, production-ready Python pipeline to detect and address skewness and outliers in numeric datasets.
It is designed to streamline the data validation process before statistical analysis or machine learning model development.

A clean, PEP8-compliant Python pipeline to:

Detect skewed featuresApply log1p transformationDefine safe value rangesValidate and log outliersGenerate visual boxplots

What Does This Code Do?

This pipeline:

Identifies highly skewed numeric features using statistical skewnessApplies np.log1p transformation to reduce skewDefines safe ranges using either percentiles or fixed bufferValidates values for each feature against defined boundsLogs outliers to CSV files (raw + sparse format)Generates visual boxplots for inspectionOutputs a summary of all validations

This helps ensure your numeric features are:

More normally distributed (for modeling),Free from extreme outliers (for analysis or visualization),Well-documented with visual and CSV-based reporting.
