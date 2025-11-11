import streamlit as st
import pandas as pd
import importlib.util
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Ad-Insights",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
def inject_custom_css():
    """Inject custom CSS for card styling and color scheme"""
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Global styles */
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Color scheme variables - Ad-Insights Brand */
        :root {
            --primary-yellow: #FFBD59;
            --yellow-light: #FFCF87;
            --yellow-lighter: #FFE7C2;
            --yellow-lightest: #FFF2DF;
            --secondary-green: #41C185;
            --secondary-blue: #458EE2;
            --text-dark: #333333;
            --text-medium: #666666;
            --text-light: #999999;
            --background-white: #FFFFFF;
            --background-light: #F5F5F5;
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }
        
        /* Main app background */
        .main {
            background-color: var(--background-light);
        }
        
        /* Card component styling */
        .card {
            background: var(--background-white);
            border-radius: 16px;
            padding: 28px;
            margin: 12px 0;
            box-shadow: var(--shadow-lg);
            border: 1px solid var(--yellow-lighter);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: linear-gradient(180deg, var(--primary-yellow) 0%, var(--secondary-green) 100%);
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-xl);
            border-color: var(--primary-yellow);
        }
        
        .card-title {
            color: var(--text-dark);
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 18px;
            letter-spacing: -0.02em;
        }
        
        /* Header styling */
        .app-header {
            text-align: center;
            background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%);
            color: var(--text-dark);
            padding: 40px 20px;
            margin: -80px -80px 40px -80px;
            border-radius: 0 0 24px 24px;
            box-shadow: var(--shadow-lg);
        }
        
        .app-header h1 {
            font-size: 2.5em;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.02em;
            text-shadow: 0 2px 4px rgba(255,255,255,0.3);
        }
        
        .app-header p {
            font-size: 1.1em;
            margin-top: 10px;
            color: var(--text-medium);
            font-weight: 500;
        }
        
        /* Login container styling */
        .login-container {
            max-width: 440px;
            margin: 80px auto;
            padding: 40px;
            background: var(--background-white);
            border-radius: 20px;
            box-shadow: var(--shadow-xl);
            border: 2px solid var(--yellow-lighter);
        }
        
        .login-header {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .login-header h1 {
            background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--secondary-green) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 2.2em;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .login-header p {
            color: var(--text-medium);
            font-size: 1em;
            font-weight: 500;
        }
        
        /* Module card styling */
        .module-card {
            background: var(--background-white);
            border-radius: 16px;
            padding: 32px 24px;
            box-shadow: var(--shadow-md);
            border: 2px solid var(--yellow-lightest);
            transition: all 0.3s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
        }
        
        .module-card:hover {
            transform: translateY(-6px);
            box-shadow: var(--shadow-xl);
            border-color: var(--primary-yellow);
            background: linear-gradient(135deg, var(--background-white) 0%, var(--yellow-lightest) 100%);
        }
        
        .module-icon {
            font-size: 3em;
            margin-bottom: 16px;
            text-align: center;
        }
        
        .module-title {
            color: var(--text-dark);
            font-size: 1.4em;
            font-weight: 700;
            margin-bottom: 12px;
            text-align: center;
        }
        
        .module-description {
            color: var(--text-medium);
            font-size: 0.95em;
            line-height: 1.6;
            margin-bottom: 20px;
            flex-grow: 1;
        }
        
        .module-features {
            color: var(--text-medium);
            font-size: 0.9em;
            line-height: 1.8;
            margin-bottom: 24px;
        }
        
        .module-features li {
            margin-bottom: 8px;
        }
        
        /* Welcome section */
        .welcome-section {
            background: linear-gradient(135deg, var(--background-white) 0%, var(--yellow-lightest) 100%);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 32px;
            box-shadow: var(--shadow-md);
            border-left: 4px solid var(--primary-yellow);
        }
        
        .welcome-section h2 {
            color: var(--text-dark);
            font-size: 1.8em;
            font-weight: 700;
            margin-bottom: 8px;
        }
        
        .welcome-section p {
            color: var(--text-medium);
            font-size: 1.05em;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: var(--background-white);
        }
        
        /* Button styling */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-yellow) 0%, var(--yellow-light) 100%);
            color: var(--text-dark);
            border: none;
            border-radius: 10px;
            padding: 12px 24px;
            font-weight: 600;
            font-size: 1em;
            transition: all 0.3s ease;
            box-shadow: var(--shadow-md);
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--yellow-light) 0%, var(--primary-yellow) 100%);
            box-shadow: var(--shadow-lg);
            transform: translateY(-2px);
        }
        
        /* Success button variant */
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, var(--secondary-green) 0%, #35a372 100%);
            color: white;
        }
        
        .stButton > button[kind="primary"]:hover {
            background: linear-gradient(135deg, #35a372 0%, var(--secondary-green) 100%);
        }
        
        /* Info styling */
        .stInfo {
            background-color: var(--yellow-lightest);
            border-left-color: var(--secondary-blue);
        }
        
        /* Success styling */
        .stSuccess {
            background-color: #e8f5f0;
            border-left-color: var(--secondary-green);
        }
        
        /* Divider styling */
        hr {
            border-color: var(--yellow-lighter);
            margin: 24px 0;
        }
        </style>
    """, unsafe_allow_html=True)

def create_card(title, content_func):
    """
    Create a styled card container
    Args:
        title: Card header text
        content_func: Function that renders card content
    """
    st.markdown(f'<div class="card">', unsafe_allow_html=True)
    if title:
        st.markdown(f'<div class="card-title">{title}</div>', unsafe_allow_html=True)
    content_func()
    st.markdown('</div>', unsafe_allow_html=True)

# Session state initialization
def initialize_session_state():
    """Initialize all session state variables"""
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if 'username' not in st.session_state:
        st.session_state['username'] = None
    
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'home'
    
    if 'uploaded_data' not in st.session_state:
        st.session_state['uploaded_data'] = None
    
    if 'data_loaded' not in st.session_state:
        st.session_state['data_loaded'] = False
    
    if 'model_config' not in st.session_state:
        st.session_state['model_config'] = {
            'target_var': None,
            'predictor_vars': [],
            'q': 0.01,
            'r': 1.0,
            'init_cov': 1.0,
            'ridge_alpha': 0.1,
            'non_negative_features': [],
            'non_positive_features': [],
            'use_log': False,
            'adaptive': False
        }
    
    if 'model_trained' not in st.session_state:
        st.session_state['model_trained'] = False
    
    if 'model_results' not in st.session_state:
        st.session_state['model_results'] = None
    
    if 'optimization_results' not in st.session_state:
        st.session_state['optimization_results'] = None
    
    if 'optimized_params' not in st.session_state:
        st.session_state['optimized_params'] = None

def navigate_to_page(page_name):
    """Navigate to a different page/module"""
    st.session_state['current_page'] = page_name
    st.rerun()

# Authentication credentials (stored in session state for persistence)
def get_credentials():
    """Get credentials from session state or initialize default"""
    if 'credentials_db' not in st.session_state:
        st.session_state.credentials_db = {
            'admin': 'admin123',
            'analyst': 'analyst123',
            'user': 'user123'
        }
    return st.session_state.credentials_db

def register_user(username, password):
    """Register a new user"""
    credentials = get_credentials()
    if username in credentials:
        return False, "Username already exists"
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    credentials[username] = password
    return True, "Account created successfully!"

def check_authentication():
    """Verify if user is authenticated"""
    return st.session_state.get('authenticated', False)

def authenticate_user(username, password):
    """
    Validate user credentials
    Args:
        username: User's username
        password: User's password
    Returns:
        bool: True if credentials are valid, False otherwise
    """
    credentials = get_credentials()
    return username in credentials and credentials[username] == password

def logout():
    """Clear session state and return to login"""
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    # Reinitialize session state
    initialize_session_state()
    st.rerun()

def show_login_page():
    """Display login form and handle authentication"""
    inject_custom_css()
    
    # Add animated login page CSS with flowing background
    st.markdown("""
        <style>
        /* Flowing gradient background animation - Subtle white tones */
        @keyframes gradientFlow {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .stApp {
            background: linear-gradient(-45deg, #FFFFFF, #FAFAFA, #F5F5F5, #FFFFFF);
            background-size: 400% 400%;
            animation: gradientFlow 20s ease infinite;
        }
        
        /* Floating particles */
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-30px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        @keyframes rotate {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
        }
        
        .login-page {
            animation: fadeIn 0.8s ease-out;
            position: relative;
        }
        
        .login-box {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-radius: 24px;
            padding: 48px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.15);
            border: 2px solid #FFBD59;
            animation: slideIn 0.6s ease-out;
            position: relative;
            z-index: 10;
        }
        
        .login-logo {
            text-align: center;
            margin-bottom: 32px;
            animation: pulse 2s ease-in-out infinite;
        }
        
        .login-logo h1 {
            background: linear-gradient(135deg, #FFBD59 0%, #41C185 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            font-size: 3.5em;
            font-weight: 800;
            margin: 0;
            letter-spacing: -2px;
        }
        
        .login-subtitle {
            color: #666666;
            font-size: 1.1em;
            text-align: center;
            margin-bottom: 32px;
            font-weight: 500;
        }
        
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        
        .shimmer-effect {
            background: linear-gradient(90deg, #FFBD59 0%, #FFCF87 50%, #FFBD59 100%);
            background-size: 1000px 100%;
            animation: shimmer 3s infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Avatar selection styles */
        .avatar-grid {
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 12px;
            margin: 16px 0;
        }
        
        .avatar-option {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            border: 3px solid transparent;
            cursor: pointer;
            transition: all 0.3s ease;
            font-size: 2em;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #F5F5F5;
        }
        
        .avatar-option:hover {
            transform: scale(1.1);
            border-color: #FFBD59;
            box-shadow: 0 4px 12px rgba(255, 189, 89, 0.4);
        }
        
        .avatar-selected {
            border-color: #41C185 !important;
            background: #FFF2DF !important;
            transform: scale(1.15);
            box-shadow: 0 6px 16px rgba(65, 193, 133, 0.4);
        }
        
        .password-strength {
            height: 4px;
            border-radius: 2px;
            margin-top: 8px;
            transition: all 0.3s ease;
        }
        
        .strength-weak { background: #FF6B6B; width: 33%; }
        .strength-medium { background: #FFBD59; width: 66%; }
        .strength-strong { background: #41C185; width: 100%; }
        
        .feature-badge {
            display: inline-block;
            background: linear-gradient(135deg, #FFBD59 0%, #FFCF87 100%);
            color: #333333;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 4px;
            box-shadow: 0 4px 12px rgba(255, 189, 89, 0.3);
        }
        
        .features-container {
            text-align: center;
            margin: 24px 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2.5, 1])
    
    with col2:
        st.markdown('<div class="login-page">', unsafe_allow_html=True)
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        
        # Logo and title
        st.markdown('''
            <div class="login-logo">
                <h1>📈 Ad-Insights</h1>
            </div>
            <div class="login-subtitle">
                Advanced Analytics & Optimization Platform
            </div>
            <div class="features-container">
                <span class="feature-badge">📊 EDA</span>
                <span class="feature-badge">🎯 Modeling</span>
                <span class="feature-badge">⚡ Optimization</span>
            </div>
        ''', unsafe_allow_html=True)
        
        # Tab selection for Login/Sign Up
        if 'show_signup' not in st.session_state:
            st.session_state.show_signup = False
        
        tab_col1, tab_col2 = st.columns(2)
        with tab_col1:
            if st.button("🔐 Login", use_container_width=True, type="primary" if not st.session_state.show_signup else "secondary"):
                st.session_state.show_signup = False
                st.rerun()
        with tab_col2:
            if st.button("✨ Sign Up", use_container_width=True, type="primary" if st.session_state.show_signup else "secondary"):
                st.session_state.show_signup = True
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if not st.session_state.show_signup:
            # Login form
            with st.form("login_form"):
                st.markdown("### Welcome Back!")
                username = st.text_input("👤 Username", placeholder="Enter your username", key="login_username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter your password", key="login_password")
                
                col_login1, col_login2 = st.columns([2, 1])
                with col_login1:
                    submit_button = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if submit_button:
                    if username and password:
                        if authenticate_user(username, password):
                            st.session_state['authenticated'] = True
                            st.session_state['username'] = username
                            st.success(f"✅ Welcome back, {username}!")
                            st.balloons()
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password. Please try again.")
                    else:
                        st.warning("⚠️ Please enter both username and password.")
            
            # Demo credentials
            with st.expander("📋 Demo Credentials"):
                st.info("""
                **Try these demo accounts:**
                - 👨‍💼 Admin: `admin` / `admin123`
                - 📊 Analyst: `analyst` / `analyst123`
                - 👤 User: `user` / `user123`
                """)
        
        else:
            # Sign up form
            st.markdown("### Create Your Account")
            
            # Avatar selection (optional)
            st.markdown("**🎨 Choose Your Avatar** (Optional)")
            avatars = ["👤", "👨", "👩", "🧑", "👨‍💼", "👩‍💼", "👨‍💻", "👩‍💻", "🧑‍💻", "👨‍🔬", "👩‍🔬", "🧑‍🔬", "👨‍🎨", "👩‍🎨", "🧑‍🎨"]
            
            # Initialize selected avatar in session state
            if 'selected_avatar' not in st.session_state:
                st.session_state.selected_avatar = "👤"
            
            # Display avatars in a grid
            avatar_cols = st.columns(5)
            for idx, avatar in enumerate(avatars):
                with avatar_cols[idx % 5]:
                    if st.button(avatar, key=f"avatar_{idx}", use_container_width=True):
                        st.session_state.selected_avatar = avatar
                        st.rerun()
            
            # Show selected avatar
            st.markdown(f"**Selected:** {st.session_state.selected_avatar}")
            st.divider()
            
            with st.form("signup_form"):
                new_username = st.text_input("👤 Choose Username", placeholder="Minimum 3 characters", key="signup_username")
                new_password = st.text_input("🔒 Choose Password", type="password", placeholder="Minimum 6 characters", key="signup_password")
                confirm_password = st.text_input("🔒 Confirm Password", type="password", placeholder="Re-enter password", key="signup_confirm")
                
                signup_button = st.form_submit_button("✨ Create Account", use_container_width=True)
                
                if signup_button:
                    if new_username and new_password and confirm_password:
                        if new_password != confirm_password:
                            st.error("❌ Passwords do not match!")
                        else:
                            success, message = register_user(new_username, new_password)
                            if success:
                                # Store avatar with user
                                if 'user_avatars' not in st.session_state:
                                    st.session_state.user_avatars = {}
                                st.session_state.user_avatars[new_username] = st.session_state.selected_avatar
                                
                                st.success(f"✅ {message}")
                                st.info(f"Your avatar: {st.session_state.selected_avatar} | You can now login!")
                                st.balloons()
                                st.session_state.show_signup = False
                                st.session_state.selected_avatar = "👤"  # Reset
                                st.rerun()
                            else:
                                st.error(f"❌ {message}")
                    else:
                        st.warning("⚠️ Please fill in all fields.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
            <div style='text-align: center; margin-top: 32px; color: #999999; font-size: 0.9em;'>
                <p>🔒 Secure • 🚀 Fast • 📊 Powerful</p>
                <p style='font-size: 0.8em;'>© 2025 Ad-Insights. All rights reserved.</p>
            </div>
        """, unsafe_allow_html=True)

def show_main_app():
    """Display the main application interface after authentication"""
    inject_custom_css()
    
    # Header
    st.markdown('''
        <div class="app-header">
            <h1>📈 Ad-Insights</h1>
            <p>Advanced Analytics & Optimization Platform</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar with user info and logout
    with st.sidebar:
        # Get user avatar
        user_avatar = "👤"
        if 'user_avatars' in st.session_state and st.session_state['username'] in st.session_state.user_avatars:
            user_avatar = st.session_state.user_avatars[st.session_state['username']]
        
        st.markdown(f"### {user_avatar} User Profile")
        st.markdown(f"**{st.session_state['username']}**")
        st.markdown("---")
        
        if st.button("🚪 Logout", use_container_width=True):
            logout()
        
        st.markdown("---")
        st.markdown("### 📌 Quick Info")
        st.info("Click on any module card below to access the analytics tools.")
        
        st.markdown("---")
        st.markdown("### 🔧 System Status")
        st.success("✓ All systems operational")
    
    # Welcome section
    st.markdown(f'''
        <div class="welcome-section">
            <h2>Welcome back, {st.session_state['username']}! 👋</h2>
            <p>Choose a module below to start your analytics journey</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Create three columns for the navigation cards
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown('''
            <div class="module-card">
                <div class="module-icon">📊</div>
                <div class="module-title">Exploratory Data Analysis</div>
                <div class="module-description">
                    Dive deep into your data with comprehensive visualization and statistical analysis tools.
                </div>
                <div class="module-features">
                    ✓ Data upload & preview<br>
                    ✓ Summary statistics<br>
                    ✓ Distribution plots<br>
                    ✓ Correlation analysis
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚀 Launch EDA Module", key="eda_btn", use_container_width=True):
            navigate_to_page('eda_projects')
    
    with col2:
        st.markdown('''
            <div class="module-card">
                <div class="module-icon">🎯</div>
                <div class="module-title">Modeling</div>
                <div class="module-description">
                    Build and train advanced Kalman Filter models with customizable parameters and constraints.
                </div>
                <div class="module-features">
                    ✓ Model configuration<br>
                    ✓ Kalman Filter training<br>
                    ✓ Performance metrics<br>
                    ✓ Coefficient tracking
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚀 Launch Modeling Module", key="modeling_btn", use_container_width=True):
            navigate_to_page('modeling')
    
    with col3:
        st.markdown('''
            <div class="module-card">
                <div class="module-icon">⚡</div>
                <div class="module-title">Optimization</div>
                <div class="module-description">
                    Automatically find the best hyperparameters and feature combinations for optimal model performance.
                </div>
                <div class="module-features">
                    ✓ Hyperparameter tuning<br>
                    ✓ Feature selection<br>
                    ✓ Grid search<br>
                    ✓ Results comparison
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚀 Launch Optimization Module", key="optimization_btn", use_container_width=True):
            navigate_to_page('optimization')
    
    # Footer section with debug info
    st.markdown("---")
    with st.expander("🔍 Session State (Debug)"):
        st.json({
            'authenticated': st.session_state['authenticated'],
            'username': st.session_state['username'],
            'data_loaded': st.session_state['data_loaded']
        })

def show_modeling_page():
    """Display the modeling page with three model type cards"""
    inject_custom_css()
    
    # Header
    st.markdown('''
        <div class="app-header">
            <h1>🎯 Modeling Hub</h1>
            <p>Choose Your Model Type</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Sidebar with navigation
    with st.sidebar:
        st.markdown("### 🏠 Navigation")
        if st.button("⬅️ Back to Home", use_container_width=True, key="back_home_modeling"):
            navigate_to_page('home')
        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state['username']}")
        if st.button("🚪 Logout", use_container_width=True, key="logout_modeling"):
            logout()
        
        st.markdown("---")
        st.markdown("### 📌 Model Types")
        st.info("Select the appropriate model type for your analysis needs.")
    
    # Welcome section
    st.markdown(f'''
        <div class="welcome-section">
            <h2>Select Your Model Type 🎯</h2>
            <p>Choose from regression, variation, ensemble, or classification models</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Create three columns for model type cards
    col1, col2, col3 = st.columns(3, gap="large")
    
    with col1:
        st.markdown('''
            <div class="module-card">
                <div class="module-icon">📈</div>
                <div class="module-title">Regression Models</div>
                <div class="module-description">
                    Traditional regression models with fixed coefficients for stable relationships.
                </div>
                <div class="module-features">
                    ✓ Linear Regression<br>
                    ✓ Ridge & Lasso<br>
                    ✓ Polynomial Regression<br>
                    ✓ Feature importance<br>
                    ✓ R² & RMSE metrics
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚀 Launch Regression Models", key="regression_btn", use_container_width=True):
            navigate_to_page('regression')
    
    with col2:
        st.markdown('''
            <div class="module-card">
                <div class="module-icon">📊</div>
                <div class="module-title">Variation Models</div>
                <div class="module-description">
                    Time-varying coefficient models like Kalman Filter where parameters evolve over time.
                </div>
                <div class="module-features">
                    ✓ Kalman Filter<br>
                    ✓ Time-varying coefficients<br>
                    ✓ Dynamic relationships<br>
                    ✓ Coefficient tracking<br>
                    ✓ Adaptive modeling
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚀 Launch Variation Models", key="variation_btn", use_container_width=True):
            navigate_to_page('variation')
    
    with col3:
        st.markdown('''
            <div class="module-card">
                <div class="module-icon">🎲</div>
                <div class="module-title">Ensemble & Classification</div>
                <div class="module-description">
                    Advanced ensemble methods and classification models for complex prediction tasks.
                </div>
                <div class="module-features">
                    ✓ Random Forest<br>
                    ✓ Gradient Boosting<br>
                    ✓ XGBoost & LightGBM<br>
                    ✓ Logistic Regression<br>
                    ✓ Multi-class support
                </div>
            </div>
        ''', unsafe_allow_html=True)
        
        if st.button("🚀 Launch Ensemble & Classification", key="ensemble_btn", use_container_width=True):
            st.info("🔗 Ensemble models app will be configured later")
    
    # Footer section
    st.markdown("---")
    st.markdown("""
        ### 💡 Model Selection Guide
        
        - **Regression Models**: Best for stable, linear relationships with fixed coefficients
        - **Variation Models**: Ideal for time-series data where relationships change over time
        - **Ensemble & Classification**: Perfect for complex patterns and categorical predictions
    """)

def load_regression_module():
    """Load and execute the Regression module"""
    inject_custom_css()
    
    # Add back button in sidebar
    with st.sidebar:
        st.markdown("### 🏠 Navigation")
        if st.button("⬅️ Back to Modeling", use_container_width=True, key="back_modeling_regression"):
            navigate_to_page('modeling')
        if st.button("🏠 Back to Home", use_container_width=True, key="back_home_regression"):
            navigate_to_page('home')
        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state['username']}")
        if st.button("🚪 Logout", use_container_width=True, key="logout_regression"):
            logout()
    
    # Import and run modelling/app.py
    try:
        # Add modelling folder to Python path
        modelling_path = Path("modelling")
        if str(modelling_path.absolute()) not in sys.path:
            sys.path.insert(0, str(modelling_path.absolute()))
        
        # Import the required modules from modelling folder
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge
        from sklearn.metrics import r2_score
        import warnings
        
        # Import modelling-specific modules
        import models
        import pipeline
        import utils as modelling_utils
        
        # Read modelling/app.py content
        regression_path = Path("modelling/app.py")
        if regression_path.exists():
            with open(regression_path, 'r', encoding='utf-8') as f:
                regression_code = f.read()
            
            # Remove or comment out st.set_page_config to avoid conflict
            regression_code = regression_code.replace('st.set_page_config(page_title="Modeling App", layout="wide")', 
                                                     '# st.set_page_config() - Handled by main app')
            
            # Execute the modified code with all necessary imports
            exec(regression_code, {
                '__name__': '__main__', 
                'st': st, 
                'pd': pd, 
                'np': np,
                'px': px,
                'go': go,
                'warnings': warnings,
                'LinearRegression': LinearRegression,
                'Ridge': Ridge,
                'Lasso': Lasso,
                'ElasticNet': ElasticNet,
                'BayesianRidge': BayesianRidge,
                'r2_score': r2_score,
                'CustomConstrainedRidge': models.CustomConstrainedRidge,
                'ConstrainedLinearRegression': models.ConstrainedLinearRegression,
                'RecursiveLeastSquaresRegressor': models.RecursiveLeastSquaresRegressor,
                'StackedInteractionModel': models.StackedInteractionModel,
                'StatsMixedEffectsModel': models.StatsMixedEffectsModel,
                'run_model_pipeline': pipeline.run_model_pipeline,
                'safe_mape': modelling_utils.safe_mape
            })
        else:
            st.error("❌ modelling/app.py file not found. Please ensure the file exists.")
            st.info("The file should be located at: " + str(regression_path.absolute()))
    except Exception as e:
        st.error(f"❌ Error loading Regression module: {str(e)}")
        st.exception(e)
        st.info("Click 'Back to Modeling' in the sidebar to return to the modeling hub.")

def load_variation_module():
    """Load and execute the Variation/Kalman Filter module"""
    inject_custom_css()
    
    # Add back button in sidebar
    with st.sidebar:
        st.markdown("### 🏠 Navigation")
        if st.button("⬅️ Back to Modeling", use_container_width=True, key="back_modeling_variation"):
            navigate_to_page('modeling')
        if st.button("🏠 Back to Home", use_container_width=True, key="back_home_variation"):
            navigate_to_page('home')
        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state['username']}")
        if st.button("🚪 Logout", use_container_width=True, key="logout_variation"):
            logout()
    
    # Import and run kalman modleling/kalman.py
    try:
        # Add kalman modeling folder to Python path
        kalman_path = Path("kalman modleling")
        if str(kalman_path.absolute()) not in sys.path:
            sys.path.insert(0, str(kalman_path.absolute()))
        
        # Import required modules
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.linear_model import Ridge
        from plotly.subplots import make_subplots
        from itertools import product
        from scipy import stats
        import io
        
        # Read kalman.py content
        kalman_file_path = Path("kalman modleling/kalman.py")
        if kalman_file_path.exists():
            with open(kalman_file_path, 'r', encoding='utf-8') as f:
                kalman_code = f.read()
            
            # Remove st.set_page_config block completely (multi-line)
            import re
            kalman_code = re.sub(
                r'st\.set_page_config\([^)]*\)',
                '# st.set_page_config removed',
                kalman_code,
                flags=re.DOTALL
            )
            
            # Execute the modified code with all necessary imports
            exec(kalman_code, {
                '__name__': '__main__',
                'st': st,
                'pd': pd,
                'np': np,
                'px': px,
                'go': go,
                'make_subplots': make_subplots,
                'StandardScaler': StandardScaler,
                'mean_absolute_error': mean_absolute_error,
                'mean_squared_error': mean_squared_error,
                'r2_score': r2_score,
                'Ridge': Ridge,
                'product': product,
                'stats': stats,
                'io': io,
                'Path': Path,
                're': re
            })
        else:
            st.error("❌ kalman modleling/kalman.py file not found. Please ensure the file exists.")
            st.info("The file should be located at: " + str(kalman_file_path.absolute()))
    except Exception as e:
        st.error(f"❌ Error loading Variation Models module: {str(e)}")
        st.exception(e)
        st.info("Click 'Back to Modeling' in the sidebar to return to the modeling hub.")

def load_optimization_module():
    """Load and execute the Optimization module"""
    inject_custom_css()
    
    # Add back button in sidebar
    with st.sidebar:
        st.markdown("### 🏠 Navigation")
        if st.button("⬅️ Back to Home", use_container_width=True, key="back_home_optimization"):
            navigate_to_page('home')
        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state['username']}")
        if st.button("🚪 Logout", use_container_width=True, key="logout_optimization"):
            logout()
    
    # Run optimiser app directly
    try:
        import sys
        import os
        
        # Change to optimiser directory temporarily
        original_dir = os.getcwd()
        optimiser_dir = Path("optimiser").absolute()
        
        # Add optimiser to path
        if str(optimiser_dir) not in sys.path:
            sys.path.insert(0, str(optimiser_dir))
        
        # Change directory to optimiser
        os.chdir(optimiser_dir)
        
        try:
            # Temporarily override st.set_page_config to prevent conflicts
            original_set_page_config = st.set_page_config
            st.set_page_config = lambda **kwargs: None  # No-op function
            
            # Import the app module
            import importlib
            app_module = None
            
            if 'app' in sys.modules:
                app_module = importlib.reload(sys.modules['app'])
            else:
                import app as app_module
            
            # Restore original set_page_config
            st.set_page_config = original_set_page_config
            
            # The app.py should run automatically when imported
            # If it has a main() function, call it
            if app_module and hasattr(app_module, 'main'):
                app_module.main()
        finally:
            # Always restore original directory
            os.chdir(original_dir)
            
    except Exception as e:
        st.error(f"❌ Error loading Optimization module: {str(e)}")
        st.exception(e)
        st.info("Click 'Back to Home' in the sidebar to return to the home page.")

def load_eda_module():
    """Load and execute the EDA module"""
    inject_custom_css()
    
    # Add back button in sidebar
    with st.sidebar:
        st.markdown("### 🏠 Navigation")
        if st.button("⬅️ Back to Home", use_container_width=True, key="back_home_eda"):
            navigate_to_page('home')
        st.markdown("---")
        st.markdown(f"**Logged in as:** {st.session_state['username']}")
        if st.button("🚪 Logout", use_container_width=True, key="logout_eda"):
            logout()
    
    # Import and run EDA.py
    try:
        # Read EDA.py content
        eda_path = Path("EDA.py")
        if eda_path.exists():
            with open(eda_path, 'r', encoding='utf-8') as f:
                eda_code = f.read()
            
            # Remove or comment out st.set_page_config to avoid conflict
            # Since page config is already set in main app
            eda_code = eda_code.replace('st.set_page_config(page_title="Meta & Bounce Rate Analytics", layout="wide")', 
                                       '# st.set_page_config() - Handled by main app')
            
            # Execute the modified code
            exec(eda_code, {'__name__': '__main__', 'st': st, 'pd': pd, 'go': go, 
                           'make_subplots': make_subplots, 'np': np, 'io': io, 
                           'datetime': datetime, 'px': px})
        else:
            st.error("❌ EDA.py file not found. Please ensure EDA.py is in the same directory.")
            st.info("The file should be located at: " + str(eda_path.absolute()))
    except Exception as e:
        st.error(f"❌ Error loading EDA module: {str(e)}")
        st.exception(e)
        st.info("Click 'Back to Home' in the sidebar to return to the main menu.")

def main():
    """Main application entry point"""
    # Initialize session state
    initialize_session_state()
    
    # Check authentication first
    if not check_authentication():
        show_login_page()
        return
    
    # Route to appropriate page based on current_page state
    current_page = st.session_state.get('current_page', 'home')
    
    if current_page == 'home':
        show_main_app()
    elif current_page == 'eda_projects':
        # Show project selection page for EDA
        from project_selector_page import show_project_selection_page
        show_project_selection_page('eda', 'EDA Insights')
    elif current_page == 'eda':
        load_eda_module()
    elif current_page == 'modeling':
        show_modeling_page()
    elif current_page == 'regression':
        load_regression_module()
    elif current_page == 'variation':
        load_variation_module()
    elif current_page == 'optimization':
        load_optimization_module()
    else:
        show_main_app()

if __name__ == "__main__":
    main()
