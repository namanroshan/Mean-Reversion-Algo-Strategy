%matplotlib inline
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import auquanToolbox.dataloader as dl

# Load the prices data for a stock
start = '2013-06-01'
end = '2016-12-31'
m= 'PG'
data = dl.load_data_nologs('nasdaq', [m], start, end)
prices = data['ADJ CLOSE']

# Compute the cumulative moving average of the price
prices['mu'] = [prices[m][:i].mean() for i in range(len(prices))]
# Plot the price and the moving average
plt.figure(figsize=(15,7))
plt.plot(prices[m])
plt.plot(prices['mu'])
plt.show()

zscores = [(prices[m][i] - prices['mu'][i]) / np.std(prices[m][:i]) for i in range(len(prices))]

# Start with no money and no positions
money = 0
count = 0
for i in range(len(prices)):
    # Sell short if the z-score is > 1
    if zscores[i] > 1:
        money += prices[m][i]
        count -= 1
    # Buy long if the z-score is < 1
    elif zscores[i] < -1:
        money -= prices[m][i]
        count += 1
    # Clear positions if the z-score between -.5 and .5
    elif abs(zscores[i]) < 0.5:
        money += count*prices[m][i]
        count = 0
print money

start = '2016-12-01'
end = '2016-12-31'
assets = ['AAPL', 'AIG', 'C', 'T', 'PG', 'JNJ', 'EOG', 'MET', 'DOW', 'AMGN']
data = dl.load_data_nologs('nasdaq', assets, start, end)
prices = data['ADJ CLOSE']
returns = prices/prices.shift(-1) -1
returns.plot(figsize=(15,7), color=['r', 'g', 'b', 'k', 'c', 'm', 'orange',
                                     'chartreuse', 'slateblue', 'silver'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.ylabel('Returns')

# Convert to numpy array to make manipulation easier
data = np.array(prices)

wreturns = (data[4] - data[0])/data[0]
# Rank securities by return, with 0 being the lowest return
order = wreturns.argsort()
ranks = order.argsort()

# For each security, take the return for the month following the first week
# Normalization for the time period doesn't matter since we're only using the returns to rank them
mreturns = (data[-1] - data[5])/data[5]
order2 = mreturns.argsort()
ranks2 = order2.argsort()

# Plot the returns for the first week vs returns for the next month to visualize them
plt.figure(figsize=(15,7))
plt.scatter(wreturns, mreturns)
plt.xlabel('Returns for the first week')
plt.ylabel('Returns for the following month');

longs = np.array([int(x < 2)for x in ranks])
shorts = np.array([int(x > 7) for x in ranks])
print 'Going long in:', [assets[i] for i in range(len(assets)) if longs[i]]
print 'Going short in:', [assets[i] for i in range(len(assets)) if shorts[i]]

# Resolve all positions and calculate how much we would have earned
print 'Yield:', sum((data[-1] - data[4])*(longs - shorts))

from statsmodels.tsa.stattools import coint
start = '2012-01-01'
end = '2016-12-31'
# Load prices data for HP and Microsoft
data = dl.load_data_nologs('nasdaq', ['MSFT', 'HP'], start, end)
X = data['ADJ CLOSE']['MSFT']
Y = data['ADJ CLOSE']['HP']
# Compute the p-value for the cointegration of the two series
_, pvalue, _ = coint(X,Y)
print pvalue

val = pd.DataFrame(index = X.index, columns=['diff','mu'])
val['diff'] = X - Y
val['mu']= [val['diff'][:i].mean() for i in range(len(val['diff']))]
plt.figure(figsize=(15,7))
plt.plot(val['diff'])
plt.plot(val['mu'])
plt.show()



mu_60d = pd.rolling_mean(val['diff'], window=90)
plt.figure(figsize=(15,7))
plt.plot(val['diff'], label='X-Y')
plt.plot(val['mu'], label='CMA')
plt.plot(mu_60d, label='60d MA')
plt.legend()

money = 0
count = 0
for i in range(len(val['diff'])):
    # Sell short if the z-score is > 1
    if zscores[i] > 1:
        money += val['diff'][i]
        count -= 1
    # Buy long if the z-score is < 1
    elif zscores[i] < -1:
        money -= val['diff'][i]
        count += 1
    # Clear positions if the z-score between -.5 and .5
    elif abs(zscores[i]) < 0.5:
        money += count*val['diff'][i]
        count = 0
        
print money

