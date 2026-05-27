# Calculate option price from Black-Scholes Model
# Using finite differences method

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# -----------------------
# Parameters
# Option parameters
r = 0.1 #interest rate
K = 100 # strike
S0 = 100 # Current price 
sigma = 0.6 # volatility
c = 0.02 # dividend yield
T = 1 # Expiry time (years)

# Grid parameters
Smax = 1000 # Max stock price
dS = 4 # grid size (stock price grio)
dt = 0.00001 # grid size (time grid)

#---------------------------------------
# Numerical option pricing using explicit 
# finite differences method
def FiniteDiff(Smax,dS,dt):
    
    # Defining grids
    S_grid = np.arange(0,Smax,dS)
    t_grid = np.arange(0,T+dt,dt)

    # Number of points
    nS = len(S_grid)
    nt = len(t_grid)
    #alpha = dt/dS**2 # Small parameter
    #print("alpha=",alpha)

    # Stores option price at various t and S
    F_grid = np.zeros([nt,nS]) 

    # Boundary Values (at t=T)
    # price at t=T is the option payoff
    F_grid[nt-1] = np.maximum(np.zeros(nS),S_grid-K) 
    
    # Calculating price backwards in time
    #start_time = time.time()
    for i in range(nt-1):

        # calculating derivatives
        dF_grid = np.diff(F_grid[nt-i-1],prepend=0.0)/dS
        ddF_grid = np.diff(dF_grid,append=1.0)/dS

        # Finite difference sequence: vector form
        G1 = 0.5*ddF_grid*(sigma*S_grid)**2
        G2 = (r-c)*S_grid*dF_grid
        G3 = (r-c)*F_grid[nt-i-1]
        F_grid[nt-i-2] = F_grid[nt-i-1] + dt* (G1+G2-G3)
    
    #print("--- %s seconds ---" % (time.time() - start_time))
    return F_grid
    
# --------------------------
# Analytical price of option
def OptionPriceAnalytical(St,t):
    t_period = T-t
    d1 = (np.log(St/K)+(r-c+0.5*sigma**2)*t_period)/(sigma*np.sqrt(t_period))
    d2 = d1-(sigma*np.sqrt(t_period))
    C = np.exp(-c*t_period)*St*norm.cdf(d1) - np.exp(-r*t_period)*K*norm.cdf(d2)
    return C

Price_grid = FiniteDiff(Smax,dS,dt)

print("Analytical price of option at S0=100:",OptionPriceAnalytical(S0,0))
print("Numerical price of option at S0=100:",Price_grid[0,int(S0/dS)])

# ------------------------------------
# Plotting options prices at different times, 
# as a function of stock price

S_grid = np.arange(0,Smax,dS)
nT = int(T/dt)

t1, t2, t3 = T, T/2, 0
n1, n2, n3 = int(t1*nT/T), int(t2*nT/T), int(t3*nT/T)

plt.plot(S_grid,Price_grid[n1], label="t=T")
plt.plot(S_grid,Price_grid[n2], label="t=T/2")
plt.plot(S_grid,Price_grid[n3], label="t=0")
plt.xlim(00,200)
plt.ylim(-3,120)
plt.xlabel("Stock price")
plt.ylabel("Option price")
plt.legend()
plt.show()