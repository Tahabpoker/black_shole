import streamlit as st
import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt

N = norm.cdf

def call_black_schole(S, K, r, T, sigma):
    """
    S: strike price
    K: current price 
    r: risk free rate of intrest
    T: time until expiration 
    sigma: annual volatility  of assets return
    """
    d1 = (np.log(S/K)+(r + sigma ** 2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S * N(d1) - K * np.exp(-r*T) * N(d2)
    return call 

def put_black_schole(S, K, r, T, sigma):
    """
    S: strike price
    K: current price 
    r: risk free rate of intrest
    T: time until expiration 
    sigma: annual volatility  of assets return
    """
    d1 = (np.log(S/K)+(r + sigma ** 2 / 2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    put = N(-d2) * K * np.exp(-r*T) - N(-d1) * S
    return put


st.title("Black-Scholes Pricing Model")

# Sidebar inputs
st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Current Asset Price (S)", min_value=0.01, value=1000.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=70.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, max_value=10.0, step=0.1)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.20, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", min_value=0.0, max_value=1.0, value=0.05, step=0.01)

# Display parameters table
st.table({
    "Current Asset Price": [S],
    "Strike Price": [K],
    "Time to Maturity": [T],
    "Volatility": [sigma],
    "Risk-Free Interest Rate": [r]
})

# Calculate option prices
calls = call_black_schole(S, K, r, T, sigma)
puts = put_black_schole(S, K, r, T, sigma)

st.markdown(
    """
    <style>
    .box {
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
    }
    .call-box {
        background-color: #28a745;
    }
    .put-box {
        background-color: #dc3545;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    st.markdown(
        f"""
        <div class="box call-box">
            <div>CALL Value</div>
            <div>${calls:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="box put-box">
            <div>PUT Value</div>
            <div>${puts:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )