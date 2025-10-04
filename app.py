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
st.table({
    "Current Asset Price": [99.93],
    "Strike Price": [70.0],
    "Time to Maturity": [1.0],
    "Volatility": [0.20],
    "Risk-Free Interest Rate": [0.05]
})

K = 100
S = 70
T = 1
sigma = 0.20
r = 0.05
# T = np.arange(0,1,0.01)

calls = call_black_schole(S,K,r,T,sigma)
puts = put_black_schole(S,K,r,T,sigma)


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
            <div >CALL Value</div>
            <div >${calls:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="box put-box">
            <div >PUT Value</div>
            <div >${puts:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True
    )



