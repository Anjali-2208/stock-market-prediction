import numpy as np
import pandas as pd
import yfinance as yf
import sqlite3
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# Initialize Database
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        conn.close()
        return True
    except:
        return False

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = c.fetchone()
    conn.close()
    return user

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.rerun()
        else:
            st.error("Invalid Credentials")

def register_page():
    st.title("Register")
    username = st.text_input("Choose a Username")
    password = st.text_input("Choose a Password", type="password")
    if st.button("Register"):
        if register_user(username, password):
            st.success("Registration Successful! Redirecting to login...")
            st.session_state["registered"] = True
            st.rerun()
        else:
            st.error("Username already exists. Try another one.")

def logout():
    st.session_state["logged_in"] = False
    st.session_state["username"] = ""
    st.rerun()

def get_custom_date():
    return st.date_input("Select a date for prediction", datetime.today())

def stock_prediction_app():
    st.sidebar.header("Stock Selection")
    stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
    st.sidebar.button("Logout", on_click=logout)
    
    model = load_model('D:\\JN\\stock predictions model.keras')
    st.header('Stock Market Predictor')
    
    custom_date = get_custom_date()
    end_date = custom_date.strftime('%Y-%m-%d')
    start_date = (custom_date - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
    
    data = yf.download(stock, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for the selected stock and date.")
        return
    
    st.subheader('Stock Data')
    st.write(data)
    
    data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
    data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])
    
    scaler = MinMaxScaler(feature_range=(0,1))
    pas_100_days = data_train.tail(100)
    data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
    data_test_scale = scaler.fit_transform(data_test)
    
    st.subheader('Price vs MA50')
    ma_50_days = data.Close.rolling(50).mean()
    fig1 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(data.Close, 'g')
    st.pyplot(fig1)
    
    st.subheader('Price vs MA50 vs MA100')
    ma_100_days = data.Close.rolling(100).mean()
    fig2 = plt.figure(figsize=(8,6))
    plt.plot(ma_50_days, 'r')
    plt.plot(ma_100_days, 'b')
    plt.plot(data.Close, 'g')
    st.pyplot(fig2)
    
    x, y = [], []
    for i in range(100, data_test_scale.shape[0]):
        x.append(data_test_scale[i-100:i])
        y.append(data_test_scale[i, 0])
    
    x, y = np.array(x), np.array(y)
    predict = model.predict(x)
    scale = 1 / scaler.scale_
    predict = predict * scale
    y = y * scale
    
    st.subheader('Original Price vs Predicted Price')
    fig3 = plt.figure(figsize=(8,6))
    plt.plot(predict, 'r', label='Predicted Price')
    plt.plot(y, 'g', label='Original Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig3)

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "registered" not in st.session_state:
    st.session_state["registered"] = False

if st.session_state["logged_in"]:
    stock_prediction_app()
elif st.session_state["registered"]:
    login_page()
else:
    st.sidebar.title("Stock Market Account")
    choice = st.sidebar.radio("Go to", ["Login", "Register", "Logout"])
    if choice == "Login":
        login_page()
    elif choice == "Register":
        register_page()
    elif choice == "Logout":
        logout()
