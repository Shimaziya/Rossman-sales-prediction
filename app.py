import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Smart Rossmann Sales Predictor",
    page_icon="🏪",
    layout="wide"
)

# -----------------------------
# Load Dataset
# -----------------------------
train = pd.read_csv("dataset/train.csv")
store = pd.read_csv("dataset/store.csv")

# Merge datasets
train = train.merge(store, on="Store", how="left")

# Convert Date
train["Date"] = pd.to_datetime(train["Date"])

# Create date features
train["Year"] = train["Date"].dt.year
train["Month"] = train["Date"].dt.month
train["Day"] = train["Date"].dt.day

# -----------------------------
# Train Model
# -----------------------------
features = [
    "Store",
    "DayOfWeek",
    "Promo",
    "SchoolHoliday",
    "Year",
    "Month",
    "Day"
]

data = train[features + ["Sales"]].sample(
    n=100000,
    random_state=42
)

X = data[features].values
y = data["Sales"].values

# Add intercept
X = np.column_stack((np.ones(len(X)), X))

# Train-test split
split = int(0.8 * len(X))

X_train = X[:split]
X_test = X[split:]
y_train = y[:split]
y_test = y[split:]

# Linear Regression using NumPy
XTX = np.dot(X_train.T, X_train)
XTy = np.dot(X_train.T, y_train)

coefficients = np.linalg.solve(XTX, XTy)

# Predictions
y_pred = np.dot(X_test, coefficients)

# Evaluation
mae = np.mean(np.abs(y_test - y_pred))
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

# -----------------------------
# Title
# -----------------------------
st.title("🏪 Smart Rossmann Store Sales Predictor")
st.write(
    "Predict store sales and compare the impact of promotional campaigns."
)

st.divider()

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Store Information")

store_id = st.sidebar.number_input(
    "Store ID",
    min_value=1,
    max_value=1115,
    value=1
)

day_of_week = st.sidebar.selectbox(
    "Day of Week",
    [1, 2, 3, 4, 5, 6, 7]
)

date = st.sidebar.date_input(
    "Date",
    value=pd.Timestamp("2015-07-01")
)

promo = st.sidebar.selectbox(
    "Promotion",
    ["No Promotion", "Promotion"]
)

school_holiday = st.sidebar.selectbox(
    "School Holiday",
    ["No", "Yes"]
)

# Convert inputs
promo_value = 1 if promo == "Promotion" else 0
holiday_value = 1 if school_holiday == "Yes" else 0

year = date.year
month = date.month
day = date.day

# -----------------------------
# Prediction Function
# -----------------------------
def predict_sales(promo_value):

    input_data = np.array([
        1,
        store_id,
        day_of_week,
        promo_value,
        holiday_value,
        year,
        month,
        day
    ])

    prediction = np.dot(input_data, coefficients)

    return max(0, prediction)

# -----------------------------
# Main Prediction
# -----------------------------
st.header("📊 Sales Prediction")

if st.button("🔮 Predict Sales", use_container_width=True):

    predicted_sales = predict_sales(promo_value)

    st.success(
        f"Predicted Sales: ₹{predicted_sales:,.2f}"
    )

    st.divider()

    # Promotion comparison
    sales_without_promo = predict_sales(0)
    sales_with_promo = predict_sales(1)

    increase = sales_with_promo - sales_without_promo

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Without Promotion",
            f"₹{sales_without_promo:,.2f}"
        )

    with col2:
        st.metric(
            "With Promotion",
            f"₹{sales_with_promo:,.2f}"
        )

    with col3:
        st.metric(
            "Expected Increase",
            f"₹{increase:,.2f}"
        )
    if increase > 0:
        increase_percent = (increase / sales_without_promo) * 100

        st.success("🎯 Promotion Recommendation")

        st.write(
            "Promotion is recommended for this store."
        )

        st.metric(
            "Expected Sales Increase",
            f"₹{increase:,.2f}",
            f"{increase_percent:.1f}%"
        )

        st.info(
            "💡 Business Insight: The model predicts higher sales "
            "when a promotion is applied."
        )

    else:
        st.warning("🎯 Promotion Recommendation")

        st.write(
            "Promotion is not recommended for this situation."
        )

        st.info(
            "💡 Business Insight: The model does not predict a "
            "significant sales benefit from promotion."
        )
# -----------------------------
# EDA & Insights
# -----------------------------

st.divider()

st.header("📈 EDA & Insights")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Sales",
        f"₹{train['Sales'].mean():,.2f}"
    )

with col2:
    st.metric(
        "Maximum Sales",
        f"₹{train['Sales'].max():,.2f}"
    )

with col3:
    st.metric(
        "Total Stores",
        f"{train['Store'].nunique()}"
    )
    

# Promotion analysis
promo_sales = train.groupby("Promo")["Sales"].mean()

st.subheader("Promotion Impact")

promo_display = pd.DataFrame({
    "Promotion": ["Without Promotion", "With Promotion"],
    "Average Sales": [
        promo_sales.get(0, 0),
        promo_sales.get(1, 0)
    ]
})

st.bar_chart(
    promo_display.set_index("Promotion")
)

# -----------------------------
# Top Stores
# -----------------------------
st.subheader("🏆 Top 5 Performing Stores")

store_sales = (
    train.groupby("Store")["Sales"]
    .mean()
    .sort_values(ascending=False)
)

st.dataframe(
    store_sales.head(5).reset_index(),
    use_container_width=True
)

# -----------------------------
# Model Performance
# -----------------------------
st.divider()

st.header("🤖 About the Model")

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE", f"{mae:,.2f}")

with col2:
    st.metric("RMSE", f"{rmse:,.2f}")

st.write(
    "The project uses Linear Regression implemented with NumPy "
    "to predict Rossmann store sales."
)

st.write(
    "The model uses store, day of week, promotion, school holiday "
    "and date-based features."
)