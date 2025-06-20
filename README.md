<b><H1>Outlier and Skewness Validation Pipeline</H1></b>

This project provides a robust, production-ready Python pipeline to detect and address skewness and outliers in numeric datasets.
It is designed to streamline the data validation process before statistical analysis or machine learning model development.

A clean, PEP8-compliant Python pipeline to:
- Detect skewed featuresApply log1p transformation
- Define safe value rangesValidate and log outliers
- Generate visual boxplots

**What Does This Code Do?**

• Identifies highly skewed numeric features using statistical skewness <br>
• Applies np.log1p transformation to reduce skew <br>
• Defines safe ranges using either percentiles or fixed buffer <br>
• Validates values for each feature against defined boundsLogs outliers to CSV files (raw + sparse format) <br>
• Generates visual boxplots for inspectionOutputs a summary of all validations <br>

This helps ensure your numeric features are: More normally distributed (for modeling),Free from extreme outliers (for analysis or visualization),Well-documented with visual and CSV-based reporting.
