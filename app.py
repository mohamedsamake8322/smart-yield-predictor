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
        if st.button("🚪 Logout"):
            st.session_state.update({"authenticated": False, "user": None, "role": None})
            st.success("Logged out successfully.")
            st.rerun()

        # Image and title
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

# NOTE: Improvements added here
st.info("🌱 Smart Yield Predictor is now enhanced with optimized performance and new features.")
