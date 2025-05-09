import streamlit as st
st.set_page_config(page_title="Smart Yield Predictor", layout="wide")
# 📦 Imports standards de Python (stdlib)
import os
import json
from datetime import datetime
from pathlib import Path
from PIL import Image
from PIL import UnidentifiedImageError
from streamlit_folium import st_folium
# 🧪 Imports de bibliothèques externes (pip install)
import bcrypt
import joblib
import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
st.title("Welcome to Smart Yield Predictor")
DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "yield_model.pkl")
PREDICTION_FILE = os.path.join(DATA_DIR, "prediction_history.csv")
USERS_FILE = os.path.join(DATA_DIR, "users.json")

os.makedirs(DATA_DIR, exist_ok=True)
# =========================================
#         INITIAL SETUP & AUTH FILE
# =========================================
USERS_FILE = "users.json"
MODEL_FILE = "yield_model.pkl"
PREDICTION_FILE = "prediction_history.csv"

if not os.path.exists(USERS_FILE):
    default_password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "78772652Moh#")
    hashed = bcrypt.hashpw(default_password.encode(), bcrypt.gensalt()).decode()
    default_users = {"admin": {"password": hashed, "role": "admin"}}
    with open(USERS_FILE, "w") as f:
        json.dump(default_users, f, indent=4)
    print(
        "✅ Default admin user created: username = 'mohamedsamake2000', password = '78772652Moh#'"
    )

# =========================================
#           AUTHENTICATION SYSTEM
# =========================================


def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {}


def authenticate(username, password):
    users = load_users()
    if username in users:
        stored_hash = users[username]["password"]
        if bcrypt.checkpw(password.encode(), stored_hash.encode()):
            return users[username]["role"]
    return None


if "authenticated" not in st.session_state:
    st.session_state.update({"authenticated": False, "user": None, "role": None})

if not st.session_state.authenticated:
    st.image("https://cdn-icons-png.flaticon.com/512/3064/3064197.png", width=100)
    st.subheader("🔐 Please log in to access the app")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        login_button = st.form_submit_button("Login")

    if login_button:
        role = authenticate(username, password)
        if role:
            st.session_state.update(
                {"authenticated": True, "user": username, "role": role}
            )
            st.success(f"✅ Welcome, {username}!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password.")
    st.stop()

# =========================================
#           SIDEBAR & LOGOUT
# =========================================

with st.sidebar:
    if st.session_state.authenticated:
        if st.button("🚪 Logout"):
            st.session_state.update(
                {"authenticated": False, "user": None, "role": None}
            )
            st.success("Logged out successfully.")
            st.rerun()

import streamlit as st
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO

# URL de l'image de plante de riz
image_url = "https://images.unsplash.com/photo-1595433562696-1c6b7c4c9b7b"

try:
    # Téléchargement de l'image depuis l'URL
    response = requests.get(image_url)

    # Vérification du type MIME pour s'assurer que c'est une image
    if "image" in response.headers.get("Content-Type", ""):
        img = Image.open(BytesIO(response.content))

        # Redimensionnement de l'image
        img = img.resize((400, 300))  # Ajustez les dimensions selon vos besoins

        # Affichage de l'image dans Streamlit
        st.image(img, caption="🌾 Plante de riz", use_column_width=False)
    else:
        st.error("Le contenu récupéré n'est pas une image.")

except UnidentifiedImageError as e:
    st.error(f"Erreur lors de l'ouverture de l'image : {e}")
except requests.RequestException as e:
    st.error(f"Erreur de téléchargement de l'image : {e}")

st.title("🌾 Smart Yield App")

# =========================================
#           ADMIN PAGE
# =========================================


def is_admin():
    return st.session_state.role == "admin"


with st.sidebar:
    if is_admin():
        admin_tab = st.selectbox("Admin Panel", ["Manage Users"])
        if admin_tab == "Manage Users":
            st.subheader("User Management")
            users = load_users()
            for user, info in users.items():
                st.write(f"**{user}** - Role: {info['role']}")
                if user != st.session_state.user and st.button(
                    f"Delete {user}", key=f"del_{user}"
                ):
                    del users[user]
                    with open(USERS_FILE, "w") as f:
                        json.dump(users, f)
                    st.success(f"User {user} deleted.")
                    st.rerun()

            st.write("### Add New User")
            with st.form("add_user_form"):
                new_username = st.text_input("Username")
                new_password = st.text_input("Password", type="password")
                new_role = st.selectbox("Role", ["user", "admin"])
                add_btn = st.form_submit_button("Add User")

            if add_btn:
                if new_username in users:
                    st.error("User already exists.")
                else:
                    hashed_password = bcrypt.hashpw(
                        new_password.encode(), bcrypt.gensalt()
                    ).decode()
                    users[new_username] = {
                        "password": hashed_password,
                        "role": new_role,
                    }
                    with open(USERS_FILE, "w") as f:
                        json.dump(users, f)
                    st.success(f"User {new_username} added.")
                    st.rerun()

# =========================================
#           HISTORY LOGIC
# =========================================

HISTORY_FILE = "detection_history.csv"


def log_detection(filename, prediction, confidence=None, user=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {
        "timestamp": timestamp,
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence if confidence else "N/A",
        "user": user if user else "anonymous",
    }

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(HISTORY_FILE, index=False)


# =========================================
#           MAIN APP LOGIC
# =========================================

menu = st.sidebar.selectbox("Navigation", ["Disease Detection", "History"])

if menu == "Disease Detection":
    st.subheader("Disease Detection")
    # Detection logic placeholder
    st.info("Detection logic to be implemented here.")

elif menu == "History":
    st.subheader("Detection History")

    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)

        # Filters
        col1, col2 = st.columns(2)

        with col1:
            disease_filter = st.selectbox(
                "Filter by prediction",
                options=["All"] + sorted(df["prediction"].unique().tolist()),
            )

        with col2:
            date_filter = st.date_input("Filter by date")

        # Apply filters
        filtered_df = df.copy()

        if disease_filter != "All":
            filtered_df = filtered_df[filtered_df["prediction"] == disease_filter]

        if date_filter:
            filtered_df["timestamp"] = pd.to_datetime(filtered_df["timestamp"])
            filtered_df = filtered_df[filtered_df["timestamp"].dt.date == date_filter]

        st.markdown("### Filtered Results")
        st.dataframe(
            filtered_df.sort_values(by="timestamp", ascending=False),
            use_container_width=True,
        )

        # Export option
        csv = filtered_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Filtered History",
            data=csv,
            file_name="filtered_detection_history.csv",
            mime="text/csv",
        )

    else:
        st.info("No detection history found yet.")

# =========================================
#         MODEL LOADING & SHAP
# =========================================

model = None
shap_enabled = False

if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        explainer = shap.Explainer(model)
        shap_enabled = True
    except Exception as e:
        st.warning(f"SHAP loading issue: {e}")

st.info(
    "🌱 Smart Yield Predictor is now enhanced with optimized performance and new features."
)

# =========================================
#           UTILITY FUNCTIONS
# =========================================


def predict_yield_model(temp, humidity, precipitation, ph, fertilizer):
    features = np.array([[temp, humidity, precipitation, ph, fertilizer]])
    prediction = model.predict(features) if model else [0.0]
    return round(prediction[0], 2), features


def explain_prediction(features):
    return explainer(features) if shap_enabled else None


def save_prediction(inputs, prediction, location=None, source="manual"):
    entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        "Temperature": inputs[0],
        "Humidity": inputs[1],
        "Precipitation": inputs[2],
        "pH": inputs[3],
        "Fertilizer": inputs[4],
        "Latitude": location[0] if location else None,
        "Longitude": location[1] if location else None,
        "Predicted_Yield": prediction,
    }
    df = pd.DataFrame([entry])
    try:
        df.to_csv(
            PREDICTION_FILE,
            mode="a",
            header=not os.path.exists(PREDICTION_FILE),
            index=False,
        )
    except Exception as e:
        st.error(f"❌ Error saving prediction: {e}")


def show_visualizations():
    if os.path.exists(PREDICTION_FILE):
        df = pd.read_csv(PREDICTION_FILE)
        st.markdown(
            f"**Average Yield:** {df['Predicted_Yield'].mean():.2f} | Max: {df['Predicted_Yield'].max():.2f} | Min: {df['Predicted_Yield'].min():.2f}"
        )
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(10, 4))
            sns.lineplot(data=df, x="Timestamp", y="Predicted_Yield")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
        with col2:
            corr = df[
                [
                    "Temperature",
                    "Humidity",
                    "Precipitation",
                    "pH",
                    "Fertilizer",
                    "Predicted_Yield",
                ]
            ].corr()
            sns.heatmap(corr, annot=True, cmap="YlGnBu")
            st.pyplot(plt.gcf())
    else:
        st.info("No prediction history found.")


# =========================================
#               MAIN UI
# =========================================

import shap
import streamlit as st
from tensorflow.keras.models import load_model

# Title and description
st.title("🌾 Smart Agricultural Yield Prediction")
st.markdown("Predict yield based on environmental and soil data with parcel location.")
st.divider()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🧬 Manual Input",
        "📍 Field Location",
        "📁 CSV Upload",
        "📊 Visualizations",
        "📥 History Export",
    ]
)

# Tab 1: Manual Input
with tab1:
    with st.form("manual_input"):
        c1, c2, c3 = st.columns(3)
        with c1:
            temp = st.slider("🌡️ Temperature (°C)", 10.0, 50.0, 25.0)
            humidity = st.slider("💧 Humidity (%)", 0.0, 100.0, 70.0)
        with c2:
            precipitation = st.slider("🌧️ Precipitation (mm)", 0.0, 300.0, 100.0)
            ph = st.slider("🧪 Soil pH", 3.5, 9.0, 6.5)
        with c3:
            fertilizer = st.number_input("🌿 Fertilizer (kg/ha)", 0.0, 500.0, 100.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        st.session_state.inputs = [temp, humidity, precipitation, ph, fertilizer]
        prediction, features = predict_yield_model(*st.session_state.inputs)
        st.metric("Estimated Yield", f"{prediction} q/ha")
        save_prediction(st.session_state.inputs, prediction)

        if shap_enabled:
            with st.expander("🔍 SHAP Explanation"):
                shap_values = explain_prediction(features)
                st.set_option("deprecation.showPyplotGlobalUse", False)
                st.pyplot(shap.plots.waterfall(shap_values[0]))
        else:
            st.info("🔎 SHAP explanations not available.")


# Function to load the model (path needs to be correct)
def load_cnn_model():
    model_path = "path_to_your_saved_model"  # Replace with your actual model path
    model = load_model(model_path)
    return model


import datetime
import os

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Function to load the model
def load_model_from_path(model_path):
    model = load_model(model_path)
    return model


# Function to make the prediction
def predict_disease(img_path, model):
    img = image.load_img(
        img_path, target_size=(224, 224)
    )  # Adjust the size to your model's input size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    prediction = model.predict(img_array)
    return prediction


# Display the image and the prediction
def display_results(image_path, prediction):
    st.image(image_path, caption="Uploaded Image", use_column_width=True)
    predicted_class = np.argmax(
        prediction, axis=-1
    )  # Assuming the model gives probabilities
    st.write(
        f"Predicted Class: {predicted_class}"
    )  # You may want to map this to actual labels
    # Optional: Display the probability if your model provides it
    st.write(f"Prediction Probability: {np.max(prediction)}")

st.title("🌿 Plant Disease Detection")
st.markdown(
    "Upload a leaf image to detect possible diseases using a deep learning model."
)

# Load the model (path to your saved model)
model = load_model_from_path("path_to_your_saved_model")  # Replace with your model path

# File uploader section
uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:

        image = "Image".open(uploaded_file)

        st.image(image, caption="Uploaded Leaf", use_column_width=True)

    except Exception as e:
        # Si une erreur survient, l'afficher
        st.error(f"❌ Error loading the image: {e}")

        st.info("Analyzing image... (model loading or training in background)")
    except Exception as e:
        st.error(f"❌ Error loading the image: {e}")

    # Prediction
    prediction = predict_disease(uploaded_file, model)

    # Display results
    display_results(uploaded_file, prediction)

    # Save detection to history (CSV example)
    history_path = "detection_history.csv"
    new_entry = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_disease": "Tomato - Late Blight",  # Replace with actual disease
        "image_name": uploaded_file.name,
    }
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
    df.to_csv(history_path, index=False)

    st.info("Detection saved to history.")
import datetime
import os
import uuid

import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Function to load the model
def load_model_from_path(model_path):
    model = load_model(model_path)
    return model


# Function to make the prediction
def predict_disease(img_path, model):
    img = image.load_img(
        img_path, target_size=(224, 224)
    )  # Adjust the size to your model's input size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    prediction = model.predict(img_array)
    return prediction


# Function to display disease information
def display_disease_info(disease_name, disease_info):
    st.write(f"**Symptoms:** {disease_info[disease_name]['Symptoms']}")
    st.write(f"**Cause:** {disease_info[disease_name]['Cause']}")
    st.write(f"**Treatment:** {disease_info[disease_name]['Treatment']}")


# Function to save prediction history
def save_detection_to_history(uploaded_file, prediction):
    history_path = "detection_history.csv"
    new_entry = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_disease": prediction,  # Replace with actual disease prediction
        "image_name": uploaded_file.name,
    }
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
    df.to_csv(history_path, index=False)


# Streamlit user interface
st.title("🌿 Plant Disease Detection")

# Load the model if it's not already loaded in session state
if "model" not in st.session_state:
    st.session_state["model"] = load_model_from_path(
        "path_to_your_saved_model"
    )  # Replace with actual model path

# Display disease information
st.subheader("📚 Disease Information")
disease_info = {
    "Tomato - Late Blight": {
        "Symptoms": "Dark, water-soaked spots on leaves and stems. White fungal growth under leaves.",
        "Cause": "Caused by the oomycete Phytophthora infestans.",
        "Treatment": "Apply fungicides early. Remove and destroy infected plants. Avoid overhead watering.",
    },
    "Corn - Leaf Spot": {
        "Symptoms": "Small, circular lesions with tan centers and dark borders.",
        "Cause": "Caused by fungal pathogens like Bipolaris spp.",
        "Treatment": "Use resistant varieties. Practice crop rotation. Apply fungicides if necessary.",
    },
}

selected_disease = st.selectbox(
    "Select a disease to learn more:", list(disease_info.keys())
)

if selected_disease:
    display_disease_info(selected_disease, disease_info)

# File uploader section
uploaded_file = st.file_uploader(
    "Upload a plant leaf image", type=["jpg", "png", "jpeg"]
)

import uuid

if uploaded_file is not None:
    # Sauvegarde temporaire de l'image téléchargée
    unique_filename = f"temp_image_{uuid.uuid4().hex}.jpg"
    with open(unique_filename, "wb") as f:
        f.write(uploaded_file.getbuffer())


    # Make the prediction
    prediction = predict_disease(unique_filename, st.session_state["model"])

    # Display the results
    predicted_class = np.argmax(prediction, axis=-1)  # Assuming the model gives probabilities
    st.write(f"Prediction: {predicted_class}")

import uuid
import streamlit as st

if uploaded_file is not None:
    try:
        # Sauvegarde temporaire de l'image téléchargée
        unique_filename = f"temp_image_{uuid.uuid4().hex}.jpg"
        with open(unique_filename, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error(f"❌ Error saving the image: {e}")
# You may want to map this to actual labels

    # Optional: Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Leaf Image", use_column_width=True)

    # Save detection to history
    save_detection_to_history(uploaded_file, predicted_class)

    st.info("Detection saved to history.")

# Optional: Display history
if st.checkbox("Show detection history"):
    if os.path.exists("detection_history.csv"):
        history_df = pd.read_csv("detection_history.csv")
        st.dataframe(history_df)
    else:
        st.info("No detection history available yet.")
import datetime

import folium
import numpy as np
import pandas as pd
import streamlit as st
from st_folium import st_folium
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


# Function to load the model
def load_model_from_path(model_path):
    model = load_model(model_path)
    return model


# Function to make the prediction
def predict_disease(img_path, model):
    img = image.load_img(
        img_path, target_size=(224, 224)
    )  # Adjust the size to your model's input size
    img_array = image.img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    prediction = model.predict(img_array)

    # Decode the prediction
    class_names = [
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Healthy",
        "Maize_Common_rust",
        # Add other disease class names here...
    ]
    predicted_class = class_names[
        np.argmax(prediction)
    ]  # Find the class with the highest probability

    return predicted_class


# Function to display disease information
def display_disease_info(disease_name, disease_info):
    st.write(f"**Symptoms:** {disease_info[disease_name]['Symptoms']}")
    st.write(f"**Cause:** {disease_info[disease_name]['Cause']}")
    st.write(f"**Treatment:** {disease_info[disease_name]['Treatment']}")


# Function to save prediction history
def save_detection_to_history(uploaded_file, prediction):
    history_path = "detection_history.csv"
    new_entry = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predicted_disease": prediction,
        "image_name": uploaded_file.name,
    }
    if os.path.exists(history_path):
        df = pd.read_csv(history_path)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])
    df.to_csv(history_path, index=False)


# Streamlit user interface
st.title("🌿 Plant Disease Detection")

# Load the model if it's not already loaded in session state
if "model" not in st.session_state:
    st.session_state["model"] = load_model_from_path(
        "path_to_your_saved_model"
    )  # Replace with actual model path

# Display disease information
st.subheader("📚 Disease Information")
disease_info = {
    "Tomato - Late Blight": {
        "Symptoms": "Dark, water-soaked spots on leaves and stems. White fungal growth under leaves.",
        "Cause": "Caused by the oomycete Phytophthora infestans.",
        "Treatment": "Apply fungicides early. Remove and destroy infected plants. Avoid overhead watering.",
    },
    "Corn - Leaf Spot": {
        "Symptoms": "Small, circular lesions with tan centers and dark borders.",
        "Cause": "Caused by fungal pathogens like Bipolaris spp.",
        "Treatment": "Use resistant varieties. Practice crop rotation. Apply fungicides if necessary.",
    },
}

selected_disease = st.selectbox(
    "Select a disease to learn more:", list(disease_info.keys())
)

if selected_disease:
    display_disease_info(selected_disease, disease_info)

# Tab for disease detection (Upload image and predict)
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🧬 Manual Input",
        "📍 Field Location",
        "📁 CSV Upload",
        "📊 Visualizations",
        "📥 History Export",
    ]
)

# Tab 1: Disease Prediction
with tab1:
    uploaded_file = st.file_uploader(
        "Upload a plant leaf image", type=["jpg", "png", "jpeg"]
    )
    if uploaded_file is not None:
        try:
            # Sauvegarde temporaire de l'image téléchargée
            unique_filename = f"temp_image_{uuid.uuid4().hex}.jpg"
            with open(unique_filename, "wb") as f:
                f.write(uploaded_file.getbuffer())
        except Exception as e:
            st.error(f"❌ Error saving the image: {e}")

        # Make the prediction
        prediction = predict_disease(unique_filename, st.session_state["model"])

        # Display the results
        display_results(unique_filename, prediction)

        # Save detection to history
        save_detection_to_history(uploaded_file, prediction)

        st.info("Detection saved to history.")

# Tab 2: Field Location
with tab2:
    st.subheader("📍 Select Field Location")
    m = folium.Map(location=[14.5, -14.5], zoom_start=6)
    result = st_folium(m, height=350, width=700, key="field_map")

    if result and result.get("last_clicked"):
        latlon = (result["last_clicked"]["lat"], result["last_clicked"]["lng"])
        st.session_state["selected_location"] = latlon
        st.success(f"Location selected: {latlon}")

    if "selected_location" in st.session_state:
        if "inputs" in st.session_state:
            if st.button("Predict with this location"):
                prediction, features = predict_yield_model(*st.session_state.inputs)
                st.metric("Estimated Yield", f"{prediction} q/ha")
                save_prediction(
                    st.session_state.inputs,
                    prediction,
                    location=st.session_state["selected_location"],
                )
        else:
            st.warning("⚠️ Please make a prediction first in the 'Manual Input' tab.")
# Tab 3: CSV Upload
with tab3:
    st.subheader("Upload CSV File")
    csv = st.file_uploader("Upload input CSV", type=["csv"])
    if csv:
        df_csv = pd.read_csv(csv)
        required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]
        if all(col in df_csv.columns for col in required_cols):
            try:
                # Prepare the data for prediction (ensure it's in the correct format)
                input_data = df_csv[required_cols].values  # Convert to NumPy array
                predictions = model.predict(input_data)
                df_csv["Predicted_Yield"] = (
                    predictions  # Add predictions as a new column
                )

                # Display and save the output
                st.dataframe(df_csv)
                df_csv.to_csv("predictions_output.csv", index=False)
                st.success("Predictions added to CSV!")
            except Exception as e:
                st.error(f"Error predicting: {e}")
        else:
            st.error("CSV missing required columns.")

# Tab 4: Visualizations
with tab4:
    if "show_visualizations" in globals():
        show_visualizations()
    else:
        st.warning("Visualization function not defined.")

# Tab 5: Export
with tab5:
    st.subheader("Export Prediction History")
    PREDICTION_FILE = "detection_history.csv"  # Ensure this file exists
    if os.path.exists(PREDICTION_FILE):
        df = pd.read_csv(PREDICTION_FILE)
        st.download_button(
            "Download Prediction History",
            df.to_csv(index=False),
            file_name="prediction_history.csv",
            mime="text/csv",
        )
    else:
        st.info("No prediction history available.")

# Footer
st.markdown("---")
st.markdown("© 2025 AgriNest • Powered by Mohamed SAMAKE • AI Agriculture")

import matplotlib.pyplot as plt

# =========================================
#   FERTILIZATION RECOMMENDATION SYSTEM
# =========================================
st.header("🧪 Fertilization Recommendation System")

crops = ["Maize", "Rice", "Wheat", "Tomato"]
soil_types = ["Clay", "Sandy", "Loamy"]

with st.form("fertilizer_form"):
    st.subheader("Enter your field parameters")
    crop = st.selectbox("Crop", crops)
    soil_type = st.selectbox("Soil Type", soil_types)
    n = st.slider("Nitrogen (N)", 0, 200, 50)
    p = st.slider("Phosphorus (P)", 0, 200, 40)
    k = st.slider("Potassium (K)", 0, 200, 40)
    ph = st.slider("Soil pH", 3.0, 9.0, 6.5)

    submit = st.form_submit_button("Get Recommendation")

if submit:
    st.subheader("📋 Recommended Fertilizer Doses")

    # Simple rule-based logic (can be improved with real agronomic data)
    recommendation = {"N": max(0, 120 - n), "P": max(0, 90 - p), "K": max(0, 100 - k)}

    st.markdown(
        f"""
    - **Crop:** {crop}  
    - **Soil Type:** {soil_type}  
    - **Soil pH:** {ph}  
    ---
    ✅ **Recommended Fertilizer Doses:**
    - Nitrogen (N): **{recommendation['N']} kg/ha**
    - Phosphorus (P): **{recommendation['P']} kg/ha**
    - Potassium (K): **{recommendation['K']} kg/ha**
    """
    )

    # Visualize fertilizer recommendations
    fig, ax = plt.subplots()
    ax.bar(
        recommendation.keys(),
        recommendation.values(),
        color=["blue", "orange", "green"],
    )
    ax.set_title("Recommended Fertilizer Doses (kg/ha)")
    ax.set_ylabel("Amount (kg/ha)")
    st.pyplot(fig)

    # Soil pH warnings
    if ph < 5.5:
        st.warning("⚠️ The soil is too acidic. Liming may be recommended.")
    elif ph > 7.5:
        st.warning(
            "⚠️ The soil is alkaline. pH correction may be required depending on the crop."
        )

    st.success("✅ Fertilizer recommendation successfully generated.")
# =========================================
#     INTERACTIVE FIELD MAP (WATER STRESS)
# =========================================
import folium
import streamlit as st
from streamlit_folium import st_folium

st.header("🗺️ Field Map & Water Stress Simulation")

with st.form("map_form"):
    st.subheader("Field Information")
    field_name = st.text_input("Field Name", "Field 1")
    region = st.text_input("Region", "Sikasso")
    latitude = st.number_input("Latitude", value=12.65)
    longitude = st.number_input("Longitude", value=-7.98)
    stress_level = st.slider("Simulated Water Stress Level", 0, 100, 30)

    map_submit = st.form_submit_button("Display Field on Map")

if map_submit:
    st.success(f"🗺️ Displaying {field_name} in {region}")

    # Create the map centered at the provided latitude and longitude
    m = folium.Map(location=[latitude, longitude], zoom_start=7)

    # Define stress color based on water stress level
    if stress_level < 30:
        color = "green"
    elif stress_level < 70:
        color = "orange"
    else:
        color = "red"

    # Popup text showing field details and stress level
    popup_text = f"""
    <b>{field_name}</b><br>
    Region: {region}<br>
    Water Stress Level: {stress_level}%
    """

    # Add marker with color based on stress level
    folium.Marker(
        location=[latitude, longitude],
        popup=popup_text,
        icon=folium.Icon(color=color, icon="leaf"),
    ).add_to(m)

    # Add circle indicating the affected area based on stress level
    folium.Circle(
        location=[latitude, longitude],
        radius=1000,  # You can adjust this based on how large you want the radius to be
        color=color,
        fill=True,
        fill_opacity=0.4,
    ).add_to(m)

    # Render the map
    st_data = st_folium(m, width=700, height=500)
if stress_level < 20:
    color = "green"
elif stress_level < 40:
    color = "lightgreen"
elif stress_level < 60:
    color = "orange"
elif stress_level < 80:
    color = "red"
else:
    color = "darkred"

import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.header("🧪 Leaf Disease Detection")
DATA_DIR = "data"
CNN_MODEL_PATH = os.path.join(DATA_DIR, "leaf_disease_model.h5")


@st.cache_resource
def load_cnn_model():
    if os.path.exists(CNN_MODEL_PATH):
        return tf.keras.models.load_model(CNN_MODEL_PATH)
    return None


cnn_model = load_cnn_model()

st.subheader("Upload a Leaf Image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file and cnn_model:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
        img = image.resize((224, 224))  # S'assurer que
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = cnn_model.predict(img_array)[0]
        classes = ["Healthy", "Disease A", "Disease B", "Disease C"]
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.success(f"🧬 Prediction: **{predicted_class}** ({confidence:.2f}%)")

    except UnidentifiedImageError:
        st.error(
            "❌ The uploaded file is not a valid image. Please upload a JPG or PNG file."
        )

    except Exception as e:
        st.error(f"❌ Unexpected error while processing the image: {e}")

        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        prediction = cnn_model.predict(img_array)[0]
        classes = ["Healthy", "Disease A", "Disease B", "Disease C"]  # À adapter
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"🧬 Prediction: **{predicted_class}** ({confidence:.2f}%)")

    except Exception as e:
        st.error(f"Error processing the image: {e}")

elif uploaded_file and not cnn_model:
    st.warning(
        "🚫 CNN model not found. Please upload 'leaf_disease_model.h5' to the 'data' folder."
    )
