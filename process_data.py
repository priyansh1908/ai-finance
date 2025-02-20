import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load Datasets
loan_df = pd.read_csv("data/loan_data.csv")
esg_df = pd.read_csv("data/esg_data.csv")

# Convert all column names to lowercase (optional but helpful)
loan_df.columns = loan_df.columns.str.lower()

# Identify numeric columns
numeric_cols = loan_df.select_dtypes(include=['number']).columns

# Fill missing values ONLY for numeric columns
loan_df[numeric_cols] = loan_df[numeric_cols].fillna(loan_df[numeric_cols].mean())

# Encode categorical columns separately
if 'purpose' in loan_df.columns:
    loan_df['purpose'] = LabelEncoder().fit_transform(loan_df['purpose'])

# Standardize numeric features
scaler = StandardScaler()
loan_df[numeric_cols] = scaler.fit_transform(loan_df[numeric_cols])

# Process ESG Data (ensure numeric processing)
esg_numeric_cols = esg_df.select_dtypes(include=['number']).columns
esg_df[esg_numeric_cols] = esg_df[esg_numeric_cols].fillna(esg_df[esg_numeric_cols].mean())

# Save Processed Data
loan_df.to_csv("data/processed_loan_data.csv", index=False)
esg_df.to_csv("data/processed_esg_data.csv", index=False)

print("✅ Data preprocessing completed successfully!")
