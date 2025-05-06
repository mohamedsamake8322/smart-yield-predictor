# =========================================
#              IMPORTS
# =========================================
import os
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

USERS_FILE = "users.json"
MODEL_FILE = "yield_model.pkl"
PREDICTION_FILE = "prediction_history.csv"

if not os.path.exists(USERS_FILE):
    default_password = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")
    hashed = bcrypt.hashpw(default_password.encode(), bcrypt.gensalt()).decode()
    default_users = {"admin": {"password": hashed, "role": "admin"}}
    with open(USERS_FILE, "w") as f:
        json.dump(default_users, f, indent=4)
    print("✅ Default admin user created: username = 'admin', password = 'admin123'")

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

# Session state init
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
        if st.button("🔚 Logout"):
            st.session_state.update({"authenticated": False, "user": None, "role": None})
            st.success("Logged out successfully.")
            st.rerun()
    st.image("https://images.unsplash.com/photo-1501004318641-b39e6451bec6", use_column_width=True)
    st.title("🌿 Smart Yield App")

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
                if st.button(f"Delete {user}", key=f"del_{user}"):
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

def display_map():
    m = folium.Map(location=[14.5, -14.5], zoom_start=5)
    return st_folium(m, height=300, width=700)

def show_visualizations():
    if os.path.exists(PREDICTION_FILE):
        df = pd.read_csv(PREDICTION_FILE)
        st.markdown(f"**Average Yield:** {df['Predicted_Yield'].mean():.2f} | Max: {df['Predicted_Yield'].max():.2f} | Min: {df['Predicted_Yield'].min():.2f}")
        col1, col2 = st.columns(2)
        with col1:
            plt.figure(figsize=(10, 4))
            sns.lineplot(data=df, x="Timestamp", y="Predicted_Yield")
            plt.xticks(rotation=45)
            st.pyplot(plt.gcf())
        with col2:
            corr = df[["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer", "Predicted_Yield"]].corr()
            sns.heatmap(corr, annot=True, cmap="YlGnBu")
            st.pyplot(plt.gcf())
    else:
        st.info("No prediction history found.")

# =========================================
#               MAIN UI
# =========================================
st.title("🌾 Smart Agricultural Yield Prediction")
st.markdown("Predict yield based on environmental and soil data with parcel location.")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🧬 Manual Input", "📍 Field Location", "📁 CSV Upload", "📊 Visualizations", "📥 History Export"
])

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
        prediction, features = predict_yield_model(temp, humidity, precipitation, ph, fertilizer)
        st.metric("Estimated Yield", f"{prediction} q/ha")
        save_prediction([temp, humidity, precipitation, ph, fertilizer], prediction)
        if shap_enabled:
            with st.expander("🔍 SHAP Explanation"):
                shap_values = explain_prediction(features)
                st.set_option("deprecation.showPyplotGlobalUse", False)
                st.pyplot(shap.plots.waterfall(shap_values[0]))

# Tab 2: Location
with tab2:
    st.subheader("Select Field Location")
    map_data = display_map()
    if map_data.get("last_clicked"):
        latlon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]
        st.success(f"Location selected: {latlon}")
        if st.button("Predict with this location"):
            prediction, features = predict_yield_model(temp, humidity, precipitation, ph, fertilizer)
            st.metric("Estimated Yield", f"{prediction} q/ha")
            save_prediction([temp, humidity, precipitation, ph, fertilizer], prediction, location=latlon)

# Tab 3: CSV Upload
with tab3:
    st.subheader("Upload CSV File")
    csv = st.file_uploader("Upload input CSV", type=["csv"])
    if csv:
        df_csv = pd.read_csv(csv)
        required_cols = ["Temperature", "Humidity", "Precipitation", "pH", "Fertilizer"]
        if all(col in df_csv.columns for col in required_cols):
            try:
                df_csv["Predicted_Yield"] = model.predict(df_csv[required_cols])
                st.dataframe(df_csv)
                if st.button("Save All"):
                    for _, row in df_csv.iterrows():
                        save_prediction(row[required_cols].tolist(), row["Predicted_Yield"], source="csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.error("CSV must contain required columns.")

# Tab 4: Visualizations
with tab4:
    show_visualizations()

# Tab 5: Export History
with tab5:
    if os.path.exists(PREDICTION_FILE):
        with open(PREDICTION_FILE, "rb") as f:
            st.download_button("📥 Download CSV", f, file_name="prediction_history.csv")
    else:
        st.info("No history available.")

# Footer
st.markdown("---")
st.markdown("© 2025 AgriNest • Powered by Mohamed SAMAKE • AI Agriculture Suite")
