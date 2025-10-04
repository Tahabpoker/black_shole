import streamlit as st
import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")

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

"""
this is for test bracnch only
"""



st.title("Black-Scholes Pricing Model")

st.sidebar.header("Input Parameters")
S = st.sidebar.number_input("Current Asset Price (S)", min_value=0.01, value=100.0, step=1.0)
K = st.sidebar.number_input("Strike Price (K)", min_value=0.01, value=70.0, step=1.0)
T = st.sidebar.number_input("Time to Maturity (Years)", min_value=0.01, value=1.0, max_value=10.0, step=0.1)
sigma = st.sidebar.number_input("Volatility (σ)", min_value=0.01, max_value=1.0, value=0.25, step=0.01)
r = st.sidebar.number_input("Risk-Free Interest Rate", min_value=0.0, max_value=1.0, value=0.10, step=0.01)


st.sidebar.header("Heatmap Parameters")
min_spot = st.sidebar.number_input("Min Spot Price", min_value=0.00,max_value=50.00, value=40.00, step=1.0)
max_spot = st.sidebar.number_input("Max Spot Price", min_value=50.00,max_value=100.00, value=50.00, step=1.0)


st.table({
    "Current Asset Price": [S],
    "Strike Price": [K],
    "Time to Maturity": [T],
    "Volatility": [sigma],
    "Risk-Free Interest Rate": [r]
})


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
    
K1 = np.linspace(min_spot,max_spot,5)
r1 = 0.05
# T = np.arange(0,1,0.01)
T1 = 1
sigma1 = np.linspace(0.1,1,5)
S1 = 99

K_i,sigma_i = np.meshgrid(K1,sigma1)

calls = call_black_schole(S,K_i,r,T,sigma_i)
puts = put_black_schole(S,K_i,r,T,sigma_i)

# plt.plot(calls, label = "call option")
# plt.plot(puts, label = "put option")
# plt.legend()
# plt.show()



print(f" calls: {calls} \n puts: {puts}")

# --- Seaborn Heatmap ---

fig , (ax1, ax2) = plt.subplots(1,2,figsize=(12, 10))



sns.heatmap(
    puts, 
    # xticklabels=K, 
    # yticklabels=np.round(sigma, 2), 
    cmap="viridis", 
    annot=True, 
    fmt=".2f", 
    cbar_kws={'label': 'Put Option Price'},
    ax=ax1
)

ax1.set_title("Black–Scholes Put Option Heatmap")
ax1.set_xlabel("Strike Price (K)")
ax1.set_ylabel("Volatility (σ)")

sns.heatmap(
    calls, 
    # xticklabels=K, 
    # yticklabels=np.round(sigma, 2), 
    cmap="viridis", 
    annot=True, 
    fmt=".2f", 
    cbar_kws={'label': 'Put Option Price'},
    ax=ax2
)
ax2.set_title("Black–Scholes Call Option Heatmap")
ax2.set_xlabel("Strike Price (K)")
ax2.set_ylabel("Volatility (σ)")


st.title("Option Price Heatmaps")
st.pyplot(fig)