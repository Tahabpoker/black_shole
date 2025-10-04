import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns

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

K = np.arange(90,100,1)
r = 0.05
# T = np.arange(0,1,0.01)
T = 1
sigma = np.arange(0.1,1,0.1)
S = 99

K_i,sigma_i = np.meshgrid(K,sigma)

calls = call_black_schole(S,K_i,r,T,sigma_i)
puts = put_black_schole(S,K_i,r,T,sigma_i)

# plt.plot(calls, label = "call option")
# plt.plot(puts, label = "put option")
# plt.legend()
# plt.show()


print(f" calls: {calls} \n puts: {puts}")

# --- Seaborn Heatmap ---
plt.figure(figsize=(8, 6))
sns.heatmap(
    puts, 
    # xticklabels=K, 
    # yticklabels=np.round(sigma, 2), 
    cmap="viridis", 
    annot=True, 
    fmt=".2f", 
    cbar_kws={'label': 'Put Option Price'}
)

plt.title("Black–Scholes Put Option Heatmap (Seaborn)")
plt.xlabel("Strike Price (K)")
plt.ylabel("Volatility (σ)")
plt.tight_layout()
plt.show()