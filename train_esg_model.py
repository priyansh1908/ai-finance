import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/esg_data.csv")

# Drop missing values
df = df.dropna(subset=["environment_score", "social_score", "governance_score", "industry", "exchange", "total_level"])

# Encode categorical variables
industry_map = {"Finance": 0, "Tech": 1, "Energy": 2, "Healthcare": 3, "Retail": 4, "Others": 5}
exchange_map = {"NYSE": 0, "NASDAQ": 1, "LSE": 2, "Other": 3}

df["industry_encoded"] = df["industry"].map(industry_map)
df["exchange_encoded"] = df["exchange"].map(exchange_map)

# Define Features & Target
X = df[["environment_score", "social_score", "governance_score", "industry_encoded", "exchange_encoded"]]
y = df["total_level"].astype("category").cat.codes  # Convert text labels to numbers

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit scaler on all 5 features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = model.predict(X_test_scaled)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save Model & Scaler
joblib.dump(model, "models/esg_model.pkl")
joblib.dump(scaler, "models/esg_scaler.pkl")

print("✅ ESG Model & Scaler Saved Successfully!")
