import pandas as pd

train = pd.read_csv("dataset/train.csv")

print(train.head())

print("Shape of dataset:", train.shape)
print("Columns:")
print(train.columns)

print("Missing values:")
print(train.isnull().sum())

print("Average Sales:", train["Sales"].mean())
print("Maximum Sales:", train["Sales"].max())
print("Minimum Sales:", train["Sales"].min())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.hist(train["Sales"], bins=50)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Number of Records")
plt.show()
print("Average Sales without Promotion:",
      train[train["Promo"] == 0]["Sales"].mean())

print("Average Sales with Promotion:",
      train[train["Promo"] == 1]["Sales"].mean())
store = pd.read_csv("dataset/store.csv")

print("Store dataset shape:", store.shape)
print("Store columns:")
print(store.columns)
train = train.merge(store, on="Store", how="left")

print("Merged dataset shape:", train.shape)
print(train.head())
train["Date"] = pd.to_datetime(train["Date"])

train["Year"] = train["Date"].dt.year
train["Month"] = train["Date"].dt.month
train["Day"] = train["Date"].dt.day

print("Date features created successfully")
print(train[["Date", "Year", "Month", "Day"]].head())
import numpy as np

# Select useful features
features = [
    "Store",
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "Year",
    "Month",
    "Day"
]

# Use a sample of 100,000 records for faster training
data = train[features + ["Sales"]].sample(
    n=100000,
    random_state=42
)

X = data[features].values
y = data["Sales"].values

# Add a column of 1s for the intercept
X = np.column_stack((np.ones(len(X)), X))

# Split data into training and testing sets
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
# Train Linear Regression model using NumPy

XTX = np.dot(X_train.T, X_train)
XTy = np.dot(X_train.T, y_train)

coefficients = np.linalg.solve(XTX, XTy)

print("Model trained successfully!")
print("Number of coefficients:", len(coefficients))
# Make predictions
y_pred = np.dot(X_test, coefficients)

print("First 10 Actual Sales:")
print(y_test[:10])

print("First 10 Predicted Sales:")
print(y_pred[:10])
# Calculate model performance

mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)
plt.figure(figsize=(8, 5))

plt.plot(y_test[:100], label="Actual Sales")
plt.plot(y_pred[:100], label="Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Records")
plt.ylabel("Sales")
plt.legend()

plt.show()
# Find best and worst performing stores

store_sales = train.groupby("Store")["Sales"].mean().sort_values(ascending=False)

print("Top 5 Best Performing Stores:")
print(store_sales.head())

print("Top 5 Worst Performing Stores:")
print(store_sales.tail())
# Top 5 best performing stores graph

top_stores = store_sales.head(5)

plt.figure(figsize=(8, 5))
plt.bar(top_stores.index.astype(str), top_stores.values)

plt.title("Top 5 Best Performing Stores")
plt.xlabel("Store")
plt.ylabel("Average Sales")

plt.show()
# Average sales by store type

store_type_sales = train.groupby("StoreType")["Sales"].mean()

print("Average Sales by Store Type:")
print(store_type_sales)

plt.figure(figsize=(8, 5))
plt.bar(store_type_sales.index, store_type_sales.values)

plt.title("Average Sales by Store Type")
plt.xlabel("Store Type")
plt.ylabel("Average Sales")

plt.show()