import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns

# Title of the app
st.title("Melbourne House Price Prediction App")

#Load the saved model
model = joblib.load('best_model_rf.pkl')

# Define the function for prediction
def predict_price(rooms, bathroom, landsize, latitude, longitude, include_garage, distance_to_city):
    features = pd.DataFrame({
        'Rooms': [rooms],
        'Bathroom': [bathroom],
        'Landsize': [landsize],
        'Lattitude': [latitude],
        'Longtitude': [longitude]
    })
    # Adjust features based on the options selected
    garage_factor = 1.05 if include_garage else 1.0
    distance_factor = max(1 - distance_to_city / 100, 0.7)  # Reduces price based on distance

    # Create an array of inputs
    features = np.array([[rooms, bathroom, landsize, latitude, longitude]])
    # Make prediction
    price_prediction = model.predict(features)
    adjusted_price = price_prediction[0] * garage_factor * distance_factor
    return adjusted_price

# Input fields for the features
rooms = st.number_input("Number of Rooms", min_value=1, max_value=10, value=3)
bathroom = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)
landsize = st.number_input("Landsize (square meters)", min_value=0, value=500)
latitude = st.number_input("Latitude", value=-37.81)
longitude = st.number_input("Longitude", value=144.96)

# Option to include a checkbox (for potential future features)
include_garage = st.checkbox("Include Garage?")
distance_to_city = st.slider("Distance to City Center (km)", min_value=0, max_value=50, value=10)
# Display the selected features
st.write("### Selected Features")
st.write(f"- **Rooms**: {rooms}")
st.write(f"- **Bathrooms**: {bathroom}")
st.write(f"- **Landsize**: {landsize} m²")
st.write(f"- **Latitude**: {latitude}")
st.write(f"- **Longitude**: {longitude}")
if include_garage:
    st.write("- **Garage**: Yes")

# Button for prediction
if st.button("Predict Price"):
    # Call the prediction function
    with st.spinner('Predicting prices...'):
        price = predict_price(rooms, bathroom, landsize, latitude, longitude, include_garage, distance_to_city)
        st.write(f"The estimated house price is: ${price:,.2f}")

# Visualization 1: Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
# Use the original dataset (or a sample) to compute correlations
original_data = pd.read_csv('melb_data.csv')
# Select the features you're interested in
selected_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude' , 'Price']
df = original_data[selected_features]
# Drop rows with missing values in the selected features
df = df.dropna()
# Generate the correlation matrix and plot
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot(plt)

# Visualization 2: Price vs Rooms Scatter Plot
st.subheader("Price vs Rooms Scatter Plot")
plt.figure(figsize=(6, 4))
sns.scatterplot(x=df['Rooms'], y=df['Price'])
plt.title('Rooms vs Predicted Price')
st.pyplot(plt)

# Visualization 3: Histogram of Predicted Price
st.subheader("Price Distribution")
price_list = [predict_price(r, bathroom, landsize, latitude, longitude, include_garage, distance_to_city)
              for r in range(1, 11)]  # Predict price for 1 to 10 rooms

plt.figure(figsize=(6, 4))
sns.histplot(price_list, bins=10, kde=True)
plt.title('Predicted Price Distribution')
st.pyplot(plt)