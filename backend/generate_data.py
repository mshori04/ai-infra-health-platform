import pandas as pd
import numpy as np

rows = []

for i in range(5000):

    cpu = np.random.normal(40, 15)
    memory = np.random.normal(50, 20)
    latency = np.random.normal(120, 40)
    errors = np.random.normal(2, 1)

    outage = 0

    # Inject anomalies occasionally
    if np.random.rand() < 0.1:
        cpu += np.random.randint(40, 60)
        memory += np.random.randint(30, 50)
        latency += np.random.randint(500, 2000)
        errors += np.random.randint(10, 50)

    # Label outage conditions
    if cpu > 85 or memory > 90 or latency > 1000:
        outage = 1

    rows.append([
        round(cpu, 2),
        round(memory, 2),
        round(latency, 2),
        round(errors, 2),
        outage
    ])

df = pd.DataFrame(rows, columns=[
    "cpu_usage",
    "memory_usage",
    "latency",
    "error_rate",
    "outage"
])

df.to_csv("data/infra_metrics.csv", index=False)

print("Dataset generated successfully")
print(df.head())