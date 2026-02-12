import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------
# 1. Load Log File
# -----------------------
log_path = "m_khutsishvil_server.log"

timestamps = []

log_pattern = re.compile(r"\[(.*?)\]")

with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        match = log_pattern.search(line)
        if match:
            timestamps.append(match.group(1))

# -----------------------
# 2. Convert to datetime
# -----------------------
df = pd.DataFrame(timestamps, columns=["timestamp"])
df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
df = df.dropna()

# Group by minute
df["minute"] = df["timestamp"].dt.floor("T")
requests_per_minute = df.groupby("minute").size().reset_index(name="count")

# -----------------------
# 3. Regression Analysis
# -----------------------
X = np.arange(len(requests_per_minute)).reshape(-1, 1)
y = requests_per_minute["count"].values

model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)

requests_per_minute["trend"] = trend
requests_per_minute["residual"] = y - trend

# -----------------------
# 4. Detect DDoS intervals
# -----------------------
threshold = requests_per_minute["residual"].std() * 3
ddos_points = requests_per_minute[requests_per_minute["residual"] > threshold]

# -----------------------
# 5. Visualization
# -----------------------
out_dir = os.path.join("task_3", "figures")
os.makedirs(out_dir, exist_ok=True)

plt.figure(figsize=(12,6))
plt.plot(requests_per_minute["minute"], y, label="Requests per minute")
plt.plot(requests_per_minute["minute"], trend, label="Regression Trend", linestyle="--")
plt.scatter(ddos_points["minute"], ddos_points["count"], color="red", label="DDoS Detected")
plt.legend()
plt.xticks(rotation=45)
plt.title("DDoS Detection using Regression Analysis")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "ddos_detection.png"), dpi=200)
plt.close()

print("Potential DDoS intervals:")
print(ddos_points["minute"])
