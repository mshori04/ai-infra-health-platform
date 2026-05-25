import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load telemetry data
df = pd.read_csv("data/infra_metrics.csv")

# Remove target column
X = df.drop("outage", axis=1)

# Train anomaly detector
model = IsolationForest(
    contamination=0.05,
    random_state=42
)

model.fit(X)

# Save model
joblib.dump(model, "models/anomaly_model.pkl")

print("Anomaly detection model trained")