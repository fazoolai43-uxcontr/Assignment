import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# Streamlit App UI
# =========================
st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📊 Customer Churn Prediction App")


# =========================
# Load and Prepare Data
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("customer_churn_dataset-training-master.csv")
    df = df.dropna()

    # Encode categorical columns automatically
    le = LabelEncoder()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = le.fit_transform(df[col])
    return df


df = load_data()

# Identify target column (change if yours is different)
target_col = "Churn"

# Split into features (X) and label (y)
X = df.drop(target_col, axis=1)
y = df[target_col]

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# =========================
# Train Model
# =========================
@st.cache_resource
def train_model():
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(x_train, y_train)
    return model


model = train_model()

# =========================
# Evaluate Model Accuracy
# =========================
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
st.sidebar.success(f"Model Accuracy: {acc * 100:.2f}%")

# =========================
# User Input Section
# =========================
st.sidebar.header("Enter Customer Details")


def user_input_features():
    inputs = {}
    for col in X.columns:
        val = st.sidebar.number_input(f"{col}", int(X[col].min()), int(X[col].max()), int(X[col].mean()))
        inputs[col] = val
    return pd.DataFrame([inputs])


input_df = user_input_features()

# =========================
# Show user input
# =========================
st.subheader("🔍 Entered Customer Information")
st.write(input_df)

# =========================
# Make Prediction
# =========================
prediction = model.predict(input_df)[0]
prediction_proba = model.predict_proba(input_df)[0]

st.subheader("🔮 Prediction Result")
if prediction == 1:
    st.error("⚠️ The customer is likely to churn.")
else:
    st.success("✅ The customer is not likely to churn.")

