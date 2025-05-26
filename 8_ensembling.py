import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("train_data.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Split data into train (70%), val (20%), test (10%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3)

# Train base models
lr = LinearRegression().fit(X_train, y_train)
xgb_model = xgb.XGBRegressor().fit(X_train, y_train)
rf = RandomForestRegressor().fit(X_train, y_train)

# Get validation predictions (meta features)
val_preds = pd.DataFrame({
    'lr': lr.predict(X_val),
    'xgb': xgb_model.predict(X_val),
    'rf': rf.predict(X_val)
})

# Get test predictions (for final model prediction)
test_preds = pd.DataFrame({
    'lr': lr.predict(X_test),
    'xgb': xgb_model.predict(X_test),
    'rf': rf.predict(X_test)
})

# Blend using a meta-model (linear regression on predictions)
blender = LinearRegression().fit(val_preds, y_val)
final_preds = blender.predict(test_preds)

# Evaluate
print("Blending MSE:", mean_squared_error(y_test, final_preds))
