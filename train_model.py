import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
np.random.seed(42)
n = 200
data = pd.DataFrame({
"Temperature": np.random.uniform(10, 45, n),
"Humidity": np.random.uniform(20, 90, n),
"Precipitation": np.random.uniform(0, 300, n),
"pH": np.random.uniform(4, 8.5, n),
"Fertilizer": np.random.uniform(10, 300, n)
})
data["Yield"] = (
data["Temperature"] * 0.2 +
data["Humidity"] * 0.1 +
data["Precipitation"] * 0.05 +
(7 - abs(data["pH"] - 7)) * 5 +
data["Fertilizer"] * 0.03 +
np.random.normal(0, 2, n) # un peu de bruit
)
X = data.drop("Yield", axis=1)
y = data["Yield"]
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)
joblib.dump(model, "yield_model.pkl")
print("✅ Model trained and saved as yield_model.pkl")

