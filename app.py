# =========================================
#         STRUCTURE & INIT FOLDERS
# =========================================
import os

DATA_DIR = "data"
MODEL_FILE = os.path.join(DATA_DIR, "yield_model.pkl")
PREDICTION_FILE = os.path.join(DATA_DIR, "prediction_history.csv")
USERS_FILE = os.path.join(DATA_DIR, "users.json")

os.makedirs(DATA_DIR, exist_ok=True)

# =========================================
#              IMPORTS
# =========================================
import json
import bcrypt
import joblib
import shap
import folium
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from streamlit_folium import st_folium

# =========================================
#         INITIAL SETUP & AUTH FILE
# =========================================
st.set_page_config(page_title="Smart Yield Predictor", layout="wide")

if not os.path.exists(USERS_FILE):
    default_password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")
    hashed = bcrypt.hashpw(default_password.encode(), bcrypt.gensalt()).decode()
    default_users = {"admin": {"password": hashed, "role": "admin"}}
    with open(USERS_FILE, "w") as f:
        json.dump(default_users, f, indent=4)
    print("Default admin user created")  # préférable pour le terminal
# ou, afficher uniquement si mode développeur activé

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
            st.session_state.update({"authenticated": True, "user": username, "role": role})
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
            st.session_state.update({"authenticated": False, "user": None, "role": None})
            st.success("Logged out successfully.")
            st.rerun()

        st.image("https://images.unsplash.com/photo-1501004318641-b39e6451bec6", use_container_width=True)
        st.title("🌾 Smart Yield App")

# =========================================
#           ADMIN PAGE
# =========================================
def is_admin():
    return st.session_state.role == 'admin'

with st.sidebar:
    if is_admin():
        admin_tab = st.selectbox("Admin Panel", ["Manage Users"])
        if admin_tab == "Manage Users":
            st.subheader("User Management")
            users = load_users()
            for user, info in users.items():
                st.write(f"**{user}** - Role: {info['role']}")
                if user != st.session_state.user and st.button(f"Delete {user}", key=f"del_{user}"):
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
                    hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                    users[new_username] = {"password": hashed_password, "role": new_role}
                    with open(USERS_FILE, "w") as f:
                        json.dump(users, f)
                    st.success(f"User {new_username} added.")
                    st.rerun()

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

st.info("🌱 Smart Yield Predictor is now enhanced with optimized performance and new features.")

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
        "Predicted_Yield": prediction
    }
    df = pd.DataFrame([entry])
    try:
        df.to_csv(PREDICTION_FILE, mode='a', header=not os.path.exists(PREDICTION_FILE), index=False)
    except Exception as e:
        st.error(f"❌ Error saving prediction: {e}")

def show_visualizations():
    if os.path.exists(PREDICTION_FILE):
        df = pd.read_csv(PREDICTION_FILE)
        st.markdown(f"**Average Yield:** {df['Predicted_Yield'].mean():.2f} | Max: {df['Predicted_Yield'].max():.2f} | Min: {df['Predicted_Yield'].min():.2f}")
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(10, 4))
            sns.lineplot(data=df, x="Timestamp", y="Predicted_Yield")
            plt.xticks(rotation=45)
        with col2:
            corr = df[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "Predicted_Yield"]].corr()
            sns.heatmap(corr, annot=True, cmap="YlGnBu")
            st.pyplot(plt.gcf())
    else:
        st.info("No prediction history found.")
# Footer
st.markdown("---")
st.markdown("© 2025 AgriNest • Powered by Mohamed SAMAKE • AI Agriculture")
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
    recommendation = {
        "N": max(0, 120 - n),
        "P": max(0, 90 - p),
        "K": max(0, 100 - k)
    }

    st.markdown(f"""
    - **Crop:** {crop}  
    - **Soil Type:** {soil_type}  
    - **Soil pH:** {ph}  
    ---
    ✅ **Recommended Fertilizer Doses:**
    - Nitrogen (N): **{recommendation['N']} kg/ha**
    - Phosphorus (P): **{recommendation['P']} kg/ha**
    - Potassium (K): **{recommendation['K']} kg/ha**
    """)

    if ph < 5.5:
        st.warning("⚠️ The soil is too acidic. Liming may be recommended.")
    elif ph > 7.5:
        st.warning("⚠️ The soil is alkaline. pH correction may be required depending on the crop.")

    st.success("✅ Fertilizer recommendation successfully generated.")
# =========================================
#     INTERACTIVE FIELD MAP (WATER STRESS)
# =========================================
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

    m = folium.Map(location=[latitude, longitude], zoom_start=7)

    # Define stress color based on level
    if stress_level < 30:
        color = "green"
    elif stress_level < 70:
        color = "orange"
    else:
        color = "red"

    popup_text = f"""
    <b>{field_name}</b><br>
    Region: {region}<br>
    Water Stress Level: {stress_level}%
    """

    folium.Marker(
        location=[latitude, longitude],
        popup=popup_text,
        icon=folium.Icon(color=color, icon="leaf")
    ).add_to(m)

    folium.Circle(
        location=[latitude, longitude],
        radius=1000,
        color=color,
        fill=True,
        fill_opacity=0.4
    ).add_to(m)

    st_data = st_folium(m, width=700, height=500)
# =========================================
#       LEAF DISEASE DETECTION (CNN)
# =========================================
from PIL import Image
import tensorflow as tf

st.header("🧪 Leaf Disease Detection")

# Placeholder model path — change with your actual model
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

        # Preprocess image (adapt based on your model input)
        img = image.resize((224, 224))
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

        # Predict
        prediction = cnn_model.predict(img_array)[0]
        classes = ["Healthy", "Disease A", "Disease B", "Disease C"]  # Modify with your real class names
        predicted_class = classes[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"🧬 Prediction: **{predicted_class}** ({confidence:.2f}%)")
    except Exception as e:
        st.error(f"Error processing the image: {e}")

elif uploaded_file and not cnn_model:
    st.warning("🚫 CNN model not found. Please upload 'leaf_disease_model.h5' to the 'data' folder.")
