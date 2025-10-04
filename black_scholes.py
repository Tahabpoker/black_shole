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

K = 100
r = 0.05
# T = np.arange(0,1,0.01)
T = 1
sigma = 0.03
S = np.arange(90,100,0.1)


calls = [ call_black_schole(s,K,r,T,sigma) for s in  S ]
puts = [ put_black_schole(s,K,r,T,sigma) for s in S ]

plt.plot(calls, label = "call option")
plt.plot(puts, label = "put option")
plt.show()


# print(f" calls: {calls} \n puts: {puts}")