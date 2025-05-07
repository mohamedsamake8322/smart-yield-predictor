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
from email.message import EmailMessage
from streamlit_folium import st_folium
import logging
import sqlite3

# =========================================
#         INITIAL SETUP & CONFIG
# =========================================
st.set_page_config(page_title="Smart Yield Predictor", layout="wide")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DB_FILE = "app_data.db"
MODEL_FILE = "yield_model.pkl"
DEFAULT_ADMIN_PASSWORD = os.environ.get("DEFAULT_ADMIN_PASSWORD", "admin123")

# =========================================
#           DATABASE SETUP
# =========================================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
                    timestamp TEXT,
                    source TEXT,
                    username TEXT,
                    temperature REAL,
                    humidity REAL,
                    precipitation REAL,
                    ph REAL,
                    fertilizer REAL,
                    latitude REAL,
                    longitude REAL,
                    predicted_yield REAL)''')
    conn.commit()
    conn.close()

    # Create default admin if not exists
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = 'admin'")
    if not c.fetchone():
        hashed = bcrypt.hashpw(DEFAULT_ADMIN_PASSWORD.encode(), bcrypt.gensalt()).decode()
        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                  ("admin", hashed, "admin"))
        conn.commit()
    conn.close()

init_db()

# =========================================
#           AUTHENTICATION SYSTEM
# =========================================
def authenticate(username, password):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT password, role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    conn.close()
    if result and bcrypt.checkpw(password.encode(), result[0].encode()):
        return result[1]
    return None

# Session state init
if "authenticated" not in st.session_state:
    st.session_state.update({"authenticated": False, "user": None, "role": None})

# Login handling
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
#        SIDEBAR NAVIGATION & LOGOUT
# =========================================
with st.sidebar:
    if st.session_state.authenticated:
        if st.button("🚪 Logout"):
            st.session_state.update({"authenticated": False, "user": None, "role": None})
            st.success("Logged out successfully.")
            st.rerun()

        st.image("https://images.unsplash.com/photo-1501004318641-b39e6451bec6", use_container_width=True)
        st.title("🌾 Smart Yield App")
        if st.session_state.role == 'admin':
            admin_tab = st.selectbox("Admin Panel", ["Manage Users"])
            if admin_tab == "Manage Users":
                st.subheader("User Management")
                conn = sqlite3.connect(DB_FILE)
                c = conn.cursor()
                c.execute("SELECT username, role FROM users")
                users = c.fetchall()
                for user, role in users:
                    st.write(f"**{user}** - Role: {role}")
                    if user != st.session_state.user and st.button(f"Delete {user}", key=f"del_{user}"):
                        c.execute("DELETE FROM users WHERE username = ?", (user,))
                        conn.commit()
                        st.success(f"User {user} deleted.")
                        st.rerun()

                st.write("### Add New User")
                with st.form("add_user_form"):
                    new_username = st.text_input("Username")
                    new_password = st.text_input("Password", type="password")
                    new_role = st.selectbox("Role", ["user", "admin"])
                    add_btn = st.form_submit_button("Add User")

                if add_btn:
                    c.execute("SELECT * FROM users WHERE username = ?", (new_username,))
                    if c.fetchone():
                        st.error("User already exists.")
                    else:
                        hashed = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
                        c.execute("INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                                  (new_username, hashed, new_role))
                        conn.commit()
                        st.success(f"User {new_username} added.")
                        st.rerun()
                conn.close()

# =========================================
#           MODEL LOADING
# =========================================
model = None
shap_enabled = False
if os.path.exists(MODEL_FILE):
    try:
        model = joblib.load(MODEL_FILE)
        explainer = shap.Explainer(model)
        shap_enabled = True
    except Exception as e:
        logging.warning(f"SHAP loading issue: {e}")

# =========================================
#               MAIN UI
# =========================================
st.title("🌾 Smart Agricultural Yield Prediction Dashboard")
st.markdown("Professional platform for yield predictions, field data, and visual analytics.")
st.divider()

# Dashboard Overview
st.markdown("### 📈 Dashboard Overview")
conn = sqlite3.connect(DB_FILE)
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM predictions")
total_preds = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM users")
total_users = c.fetchone()[0]
st.metric("Total Predictions", total_preds)
st.metric("Total Users", total_users)
conn.close()

# =========================================
# ADD INTERACTIVE MAP
# =========================================
st.markdown("### 🌍 Interactive Map")
# Example: Plotting a location on the map
m = folium.Map(location=[45.5236, -122.6750], zoom_start=12)  # Example coordinates
folium.Marker([45.5236, -122.6750], popup="Field Location").add_to(m)
st_folium(m, width=700, height=500)

# =========================================
#           FOOTER
# =========================================
st.markdown("---")
st.markdown("© 2025 AgriNest • Powered by Mohamed SAMAKE • AI Agriculture Suite")
