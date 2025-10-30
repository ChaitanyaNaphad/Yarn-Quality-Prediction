import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

# -----------------------------
# Load dataset and train model
# -----------------------------
file_path = "dataset.csv"
df = pd.read_csv(file_path)

# Separate input and output
X = df.drop(columns=["Yarn_Quality_Rating"])
y = df["Yarn_Quality_Rating"]

# Encode categorical feature
encoder = LabelEncoder()
X["Cotton_Type"] = encoder.fit_transform(X["Cotton_Type"])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Gradient Boosting model
model = GradientBoostingRegressor(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🧵 Yarn Quality Prediction")

st.markdown("### Enter Machine Parameters Below:")

# Input fields
cotton_type = st.selectbox("Cotton Type", ["Combed", "Carded", "Organic"])
spindle_speed = st.number_input("Spindle Speed (RPM)", min_value=5000.0, max_value=15000.0, value=10000.0)
draft_ratio = st.number_input("Draft Ratio", min_value=10.0, max_value=60.0, value=30.0)
twist_per_inch = st.number_input("Twist per Inch (TPI)", min_value=5.0, max_value=25.0, value=15.0)
traveler_weight = st.number_input("Traveler Weight (mg)", min_value=10.0, max_value=50.0, value=30.0)
roller_pressure = st.number_input("Roller Pressure (N)", min_value=5.0, max_value=25.0, value=15.0)
room_temp = st.number_input("Room Temperature (°C)", min_value=20.0, max_value=35.0, value=25.0)
humidity = st.number_input("Humidity (%RH)", min_value=40.0, max_value=80.0, value=60.0)

# When button is clicked
if st.button("🔍 Predict Yarn Quality"):
    # Encode cotton type
    cotton_encoded = encoder.transform([cotton_type])[0]

    # Prepare input data
    input_data = pd.DataFrame([[
        cotton_encoded,
        spindle_speed,
        draft_ratio,
        twist_per_inch,
        traveler_weight,
        roller_pressure,
        room_temp,
        humidity
    ]], columns=X.columns)

    # Predict
    predicted_quality = model.predict(input_data)[0]

    # Display result
    st.subheader("✅ Prediction Result:")
    st.write(f"**Predicted Yarn Quality Rating:** {round(predicted_quality, 2)} / 100")

    # Optional: Display entered data summary
    st.markdown("### Entered Parameters")
    st.dataframe(input_data)

