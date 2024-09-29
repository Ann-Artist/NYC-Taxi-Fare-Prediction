import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import json

# Load the trained scaler and XGBoost model
def load_data():
    try:
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model('xgb_model.json')
        return scaler, xgb_model
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Load the list of locations from the JSON file
def load_locations():
    try:
        with open('locations.json', 'r') as locations_file:
            locations = json.load(locations_file)
        return locations
    except Exception as e:
        st.error(f"Error loading locations: {e}")

# Function to calculate distance
def haversine_dist(lat1, lon1, lat2, lon2):
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return 6371 * c

# Create a Streamlit app
def main():
    st.markdown('<h1 style="color:red;">New York City Taxi Fare Prediction</h1>', unsafe_allow_html=True)

    # Load the data
    scaler, xgb_model = load_data()
    locations = load_locations()

    # Display available locations
    st.markdown('<h5>Available Locations:</h5>', unsafe_allow_html=True)
    location_list = [location['name'] for location in locations]
    location_dict = {i: location for i, location in enumerate(locations)}

    # User input for pickup and dropoff locations
    pickup_loc = st.selectbox("Select pickup location", location_list)
    dropoff_loc = st.selectbox("Select dropoff location", location_list)
    passenger_count = st.number_input("Enter passenger count", min_value=1, max_value=4, value=1)

    # Get the selected locations' coordinates
    pickup_latitude = location_dict[location_list.index(pickup_loc)]['latitude']
    pickup_longitude = location_dict[location_list.index(pickup_loc)]['longitude']
    dropoff_latitude = location_dict[location_list.index(dropoff_loc)]['latitude']
    dropoff_longitude = location_dict[location_list.index(dropoff_loc)]['longitude']

    # Calculate the distance for the user input
    distance = haversine_dist(pickup_latitude, pickup_longitude, dropoff_latitude, dropoff_longitude)

    # Create a DataFrame for the user input
    user_input = pd.DataFrame([[distance, passenger_count]], columns=['distance', 'passenger_count'])

    # Scale the user input
    user_input_scaled = scaler.transform(user_input)

    # Predict the fare for the user input using XGBoost
    predicted_fare = xgb_model.predict(user_input_scaled)

    # Button to show the predicted fare

    if st.button("Get Predicted Fare"):
        st.markdown(f'<h4>Predicted Taxi Fare: <span style="color:red;">${predicted_fare[0]:.2f}</span></h4>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()