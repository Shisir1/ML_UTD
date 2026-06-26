from ucimlrepo import fetch_ucirepo 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# fetch dataset 
stock_portfolio_performance = fetch_ucirepo(id=390) 
  
# data (as pandas dataframes) 
X = stock_portfolio_performance.data.features 
y = stock_portfolio_performance.data.targets 

#let's copy the data so that we don't modify the original data
X = X.copy()
y = y.copy()

# # metadata 
# print(stock_portfolio_performance.metadata) 
  
# # variable information 
#print(stock_portfolio_performance.variables) 

# print(X.head())

# print(y.head())

# print(X.shape)

print(X.info())