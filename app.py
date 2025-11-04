import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Gold Price Predictor",
    page_icon="💰",
    layout="centered"
)


st.markdown("""
    <style>
    body {
        background-color: #0d0d0d;
        color: #FFD700;
    }
    .stApp {
        background-color: #0d0d0d;
        color: #FFD700;
    }
    h1, h2, h3 {
        color: #FFD700;
        text-align: center;
    }
    .css-1d391kg {
        background-color: #0d0d0d;
    }
    </style>
""", unsafe_allow_html=True)


st.markdown("<h1>💰 Gold Price Predictor</h1>", unsafe_allow_html=True)
st.write("Enter a future year to see the predicted gold price.")


@st.cache_data
def load_data():
    df = yf.download("GC=F", start="2000-01-01")
    df.reset_index(inplace=True)
    df = df[['Date', 'Close']]
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)  # Remove timezone
    df.sort_values('Date', inplace=True)
    return df

df = load_data()


df['Days'] = (df['Date'] - df['Date'].min()).dt.days


poly_degree = 3
poly = PolynomialFeatures(degree=poly_degree)
X_poly = poly.fit_transform(df[['Days']])

model = LinearRegression()
model.fit(X_poly, df['Close'])


future_year = st.number_input("Enter future year:", min_value=2025, max_value=2100, step=1)

if st.button("Predict Price"):
    
    future_date = datetime.datetime(future_year, 1, 1)
    future_days = (future_date - df['Date'].min()).days

    
    future_price = float(model.predict(poly.transform([[future_days]]))[0])

    st.markdown(
        f"<h2>📅 Predicted Gold Price in {future_year}: ${future_price:,.2f}</h2>",
        unsafe_allow_html=True
    )

    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df['Date'], df['Close'], label="Historical Price", color="#FFD700")
    
    
    future_range = pd.date_range(df['Date'].min(), future_date, freq='M')
    future_days_range = (future_range - df['Date'].min()).days
    future_preds = model.predict(poly.transform(np.array(future_days_range).reshape(-1, 1)))

    
    future_preds = [float(p) for p in future_preds]
    
    ax.plot(future_range, future_preds, linestyle="--", color="cyan", label="Prediction")
    ax.set_xlabel("Year", color="white")
    ax.set_ylabel("Gold Price (USD)", color="white")
    ax.legend()
    st.pyplot(fig)



with st.expander("📊 View Historical Data"):
    st.dataframe(df)
