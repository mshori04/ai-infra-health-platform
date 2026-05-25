import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load dataset
df = pd.read_csv("data/infra_metrics.csv")

# Features and target
X = df.drop("outage", axis=1)
y = df["outage"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "models/outage_model.pkl")

print("Model saved successfully")