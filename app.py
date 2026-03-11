import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

st.title("Delhi Weather Analysis App")

# Load data
train = pd.read_csv("DailyDelhiClimateTrain.csv")
test = pd.read_csv("DailyDelhiClimateTest.csv")

train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])

st.subheader("Training Data")
st.write(train.head())

# Plot temperature
st.subheader("Temperature Trend")

fig, ax = plt.subplots()
ax.plot(train['date'], train['meantemp'])
ax.set_xlabel("Date")
ax.set_ylabel("Mean Temperature")

st.pyplot(fig)

# Features and target
X_train = train[['humidity', 'wind_speed', 'meanpressure']]
y_train = train['meantemp']

X_test = test[['humidity', 'wind_speed', 'meanpressure']]
y_test = test['meantemp']

# Model
model = RandomForestRegressor()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

# Metrics
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

st.subheader("Model Performance")

st.write("R2 Score:", r2)
st.write("Mean Absolute Error:", mae)

# Prediction section
st.subheader("Predict Temperature")

humidity = st.slider("Humidity", 0, 100, 50)
wind_speed = st.slider("Wind Speed", 0, 20, 5)
pressure = st.slider("Pressure", 990, 1050, 1010)

if st.button("Predict"):
    pred = model.predict([[humidity, wind_speed, pressure]])
    st.success(f"Predicted Temperature: {pred[0]:.2f} °C")