"""


#################33
Real function 
def get_sensor_data():
    ref = db.reference('sensors')
    sensor_data = ref.get()  # Fetch data from Firebase
    if sensor_data is None:
        st.warning("No sensor data found in the database.")
    return sensor_data or {}


"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import json
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from io import BytesIO
import requests
import firebase_admin
from firebase_admin import credentials, db
import time
import openai
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Hibiscus_Curly_Leaves', 'Hibiscus_Healthy', 'Hibiscus_Yellowish_leaves', 'Mango_Anthracnose',
    'Mango_Bacterial_Canker', 'Mango_Die_Black', 'Mango_Healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Peepal_Bacterial_Leaf_Spot', 'Peepal_Healthy', 'Peepal_Yellowish_leaf',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]



st.markdown(
    """
    <style>
    /* Set the background of the entire page to white */
    .stMain, .stAppHeader {
        background-color: white !important;
        color: black;  /* Set text color to black */
    }
    .stAppHeader{
        border-bottom: 1px solid #00000042;
    }

    /* Ensure buttons have a dark background for visibility */
    .stButton button {
        background-color: #444444;
        color: white;
    }

    /* Set the sidebar background to white */
    .stSidebar {
        background-color: white;
    }

    /* Set the color for DataFrame text and background */
    .stDataFrame {
        color: black !important;
        background-color: white !important;
    }

    /* Override Streamlit's default block elements to ensure text is black */
    h1, h2, h3, h4, h5, h6, p, div {
        color: black !important;
    }
    .st-emotion-cache-9ycgxx{
        color: white !important;
    }
    .st-emotion-cache-12118b6 {
        color: #f5f5f5 !important;
    }
    .st-emotion-cache-6qob1r{
        border-right: 1px solid #00000042;
    }
    </style>
    """,
    unsafe_allow_html=True
)

openai.api_key = openai_api_key


# Get location using JavaScript
def get_user_location():
    st.markdown(
        """
        <script>
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const latitude = position.coords.latitude;
                const longitude = position.coords.longitude;
                document.getElementById("latitude").value = latitude;
                document.getElementById("longitude").value = longitude;
            }
        );
        </script>
        <input type="hidden" id="latitude" name="latitude">
        <input type="hidden" id="longitude" name="longitude">
        """,
        unsafe_allow_html=True,
    )

    lat = st.text_input("Latitude", key="latitude")
    lon = st.text_input("Longitude", key="longitude")
    return lat, lon

@st.cache_resource
def get_weather_data(lat, lon, api_key):
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check for HTTP errors
        return response.json()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {http_err}")
    except Exception as err:
        st.error(f"Other error occurred: {err}")
    return None

def predict_rain(weather_info):
    try:
        # Make a POST request to the Gemini API for prediction
        url = "https://api.gemini.ai/predict"
          # This is a placeholder URL. Use the correct endpoint from Gemini API.
        headers = {
            "Authorization": f"Bearer {gemini_api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "prompt": f"Based on the following weather information, predict whether it will rain or not:\n\n{weather_info}"
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['prediction'].strip()
        else:
            st.error(f"Error in Gemini API: {response.json()['message']}")
            return "Unable to predict"
    except Exception as e:
        st.error(f"Error in Gemini API: {e}")
        return "Unable to predict"

# Function to make predictions using the pre-trained model
def model_prediction(test_image_path):
    # Load the model only once and cache it
    @st.cache_resource
    def load_model():
        return tf.keras.models.load_model("trained_plant_disease_model.keras")
    model = load_model()
    
    img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0)  # Expand dimensions to make it batch-like
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Class labels for prediction output
class_name = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Hibiscus_Curly_Leaves', 'Hibiscus_Healthy', 'Hibiscus_Yellowish_leaves', 'Mango_Anthracnose',
    'Mango_Bacterial_Canker', 'Mango_Die_Black', 'Mango_Healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Peepal_Bacterial_Leaf_Spot', 'Peepal_Healthy', 'Peepal_Yellowish_leaf',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch',
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# Initialize Firebase Admin SDK
if not firebase_admin._apps:  # Check if Firebase app is already initialized
    cred = credentials.Certificate("firebase/agribot-1fbbc-firebase-adminsdk-tugam-6a007f086c.json")  # Replace with your file path
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://agribot-1fbbc-default-rtdb.firebaseio.com/'  # Replace with your Firebase database URL
    })

# Function to fetch sensor data from Firebase
#Testing Function
# def get_sensor_data():
#     # Simulating sensor data; replace this with your actual sensor fetching logic
#     return {
#         'soilMoisture': 1000 + (5 * np.random.randn()),  # Example sensor values
#         'temperature': 25 + (2 * np.random.randn()),
#         'humidity': 60 + (5 * np.random.randn()),
#         'distance': 100 + (5 * np.random.randn())
#     }

def get_sensor_data():
    ref = db.reference('sensors')
    sensor_data = ref.get()  # Fetch data from Firebase
    if sensor_data is None:
        st.warning("No sensor data found in the database.")
    return sensor_data or {}

# Streamlit app setup
st.title("Agribot Dashboard")
st.markdown("""This dashboard monitors sensor data, predicts plant diseases using machine learning, and controls irrigation based on environmental conditions.""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Sensor Data", "Plant Disease Detection", "Irrigation Control"])

# Plant Disease Detection Page
if page == "Plant Disease Detection":
    st.header("Disease Recognition")

    # Option to upload an image or capture from the Raspberry Pi camera
    option = st.radio("Choose Image Source", ('Upload Image', 'Capture from Pi Camera'))

    # Initialize session state variables
    if 'captured_image_path' not in st.session_state:
        st.session_state.captured_image_path = None

    if option == 'Upload Image':
        # Handle image upload
        test_image = st.file_uploader("Choose an Image:")
        if test_image is not None:
            st.image(test_image, width=400, use_column_width=True)
            # Save the uploaded image to session state
            with open("uploaded_image.jpg", "wb") as f:
                f.write(test_image.getvalue())
            st.session_state.captured_image_path = "uploaded_image.jpg"

        # Predict button for uploaded image
        if st.session_state.captured_image_path and st.button("Predict Uploaded Image", key="predict_uploaded"):
            try:
                st.snow()
                st.write("Predicting the uploaded image...")
                # Perform prediction
                result_index = model_prediction(st.session_state.captured_image_path)
                st.success(f"Model is predicting it's a {class_name[result_index]}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

    elif option == 'Capture from Pi Camera':
        # Handle image capture from Raspberry Pi camera
        if 'captured_image' not in st.session_state:
            st.session_state.captured_image = None

        if st.button("Capture Image from Pi Camera"):
            try:
                response = requests.get("http://192.168.80.84:8080/capture", timeout=10)
                
                # Debugging step: print the response content
                st.write(response.content[:100])  # Show the first 100 bytes for inspection
                
                # If the content looks like an image, proceed
                img = Image.open(BytesIO(response.content))
                st.image(img, caption='Captured Image from Pi', use_column_width=True)

                # Save the captured image to session state
                st.session_state.captured_image = "captured_image.jpg"
                with open("captured_image.jpg", "wb") as f:
                    f.write(response.content)
                
                st.success("Image captured successfully.")
            except Exception as e:
                st.error(f"Error capturing image: {e}")
        # Predict button for captured image
        if st.session_state.captured_image:
            if st.button("Predict Captured Image"):
                try:
                    st.snow()
                    st.write("Predicting the captured image...")
                    # Perform prediction
                    result_index = model_prediction("captured_image.jpg")
                    st.success(f"Model is predicting it's a {class_name[result_index]}")
                except Exception as e:
                    st.error(f"Error during prediction: {e}")

# Sensor Data Page
# Page for Live Sensor Data Monitoring
soil_moisture_values = []
temperature_values = []
humidity_values = []
distance_values = []
if page == "Sensor Data":
    st.header("Live Sensor Data Monitoring")

    # Create a placeholder for the data table and plot
    data_placeholder = st.empty()
    plot_placeholder = st.empty()
    
    auto_refresh = st.checkbox("Enable Auto-Refresh")

    if auto_refresh:
        while True:
            # Fetch the latest sensor data
            sensor_data = get_sensor_data()

            # Append new values, maintaining only the last 10
            if len(soil_moisture_values) >= 10:
                soil_moisture_values.pop(0)
            if len(temperature_values) >= 10:
                temperature_values.pop(0)
            if len(humidity_values) >= 10:
                humidity_values.pop(0)
            if len(distance_values) >= 10:
                distance_values.pop(0)

            soil_moisture_values.append(sensor_data.get('soilMoisture'))
            temperature_values.append(sensor_data.get('temperature'))
            humidity_values.append(sensor_data.get('humidity'))
            distance_values.append(sensor_data.get('distance'))

            # Create DataFrame with the latest data
            data = {
                'Soil Moisture': [soil_moisture_values[-1]],
                'Temperature (°C)': [temperature_values[-1]],
                'Humidity (%)': [humidity_values[-1]],
                'Ultrasonic Distance (cm)': [distance_values[-1]]
            }
            df = pd.DataFrame(data)

            # Update the data table
            with data_placeholder.container():
                st.subheader("Latest Sensor Data")
                st.dataframe(df)

            # Plot the latest 10 sensor values
            with plot_placeholder.container():
                fig, ax = plt.subplots(4, 1, figsize=(8, 10))

                ax[0].plot(soil_moisture_values, color='green', marker='o')
                ax[0].set_title('Soil Moisture')
                ax[0].set_ylabel('Moisture (%)')

                ax[1].plot(temperature_values, color='red', marker='o')
                ax[1].set_title('Temperature (°C)')
                ax[1].set_ylabel('Temperature (°C)')

                ax[2].plot(humidity_values, color='blue', marker='o')
                ax[2].set_title('Humidity (%)')
                ax[2].set_ylabel('Humidity (%)')

                ax[3].plot(distance_values, color='purple', marker='o')
                ax[3].set_title('Ultrasonic Distance (cm)')
                ax[3].set_ylabel('Distance (cm)')

                # Adjust layout to prevent overlapping titles
                plt.tight_layout()
                st.pyplot(fig)

            # Wait for 2 seconds before updating again
            time.sleep(2)
# Irrigation Control Page
# Agribot Movement Control Page
elif page == "Irrigation Control":

    # weather_api_key = "907dbb6c6f7af0481bec1fc1c0ebc067"
    # openai_api_key = "sk-zc6yb5Jh_QPV2M99r_giLoWFI15W5q4zBkSBg5FdNgT3BlbkFJJadJSdh0eayb5wC0WGy0x5VNV0RNMD7Q2FkHRo3isA"

    st.header("Agribot Movement Control with Weather Insights")
    
    # Fetch user's location when the page is selected
    latitude, longitude = get_user_location()
    
    # Ensure latitude and longitude are valid before calling the weather API
    if latitude and longitude and latitude.strip() and longitude.strip():
        weather_data = get_weather_data(latitude, longitude, weather_api_key)

        # Check if weather_data is available
        if weather_data:
            st.write(f"Current Temperature: {weather_data['main']['temp']}°C")
            st.write(f"Humidity: {weather_data['main']['humidity']}%")
            st.write(f"Weather Condition: {weather_data['weather'][0]['description']}")
            
            # Use OpenAI to predict rain based on weather data
            weather_info = f"Temperature: {weather_data['main']['temp']}°C, Humidity: {weather_data['main']['humidity']}%, Condition: {weather_data['weather'][0]['description']}"
            rain_prediction = predict_rain(weather_info)
            st.write(f"Rain Prediction: {rain_prediction}")
        else:
            st.warning("Weather data is unavailable. Please ensure that location services are enabled.")
    else:
        st.warning("Location data not available. Please allow access to your location.")
    
    # Function to send movement commands to Firebase
    def send_movement_command(command):
        try:
            ref = db.reference('control/movement')  # Change the Firebase reference path
            ref.set(command)
            st.success(f"Movement command '{command}' sent successfully.")
        except Exception as e:
            st.error(f"Error sending movement command: {e}")

    # Movement control buttons
    if st.button("Move Forward"):
        send_movement_command("FORWARD")

    if st.button("Move Backward"):
        send_movement_command("BACKWARD")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Turn Left"):
            send_movement_command("LEFT")

    with col2:
        if st.button("Turn Right"):
            send_movement_command("RIGHT")

    if st.button("Stop"):
        send_movement_command("STOP")
