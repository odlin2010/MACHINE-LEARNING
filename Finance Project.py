
# coding: utf-8

# # Odlin Montila

# # Financial Data Project - 
# 
# **In this data project i focus on exploratory data analysis of stock prices.i will be focusing on bank stocks and see how they progressed throughout the [financial crisis](https://en.wikipedia.org/wiki/Financial_crisis_of_2007%E2%80%9308) all the way to early 2016.**

# In[1]:


from pandas_datareader import data, wb
import pandas as pd
import numpy as np
import datetime
get_ipython().magic('matplotlib inline')


# In[2]:


start = datetime.datetime(2006, 1, 1)
end = datetime.datetime(2016, 1, 1)


# In[3]:


# Bank of America
BAC = data.DataReader("BAC", 'google', start, end)

# CitiGroup
C = data.DataReader("C", 'google', start, end)

# Goldman Sachs
GS = data.DataReader("GS", 'google', start, end)

# JPMorgan Chase
JPM = data.DataReader("JPM", 'google', start, end)

# Morgan Stanley
MS = data.DataReader("MS", 'google', start, end)

# Wells Fargo
WFC = data.DataReader("WFC", 'google', start, end)


# In[4]:


# Could also do this for a Panel Object
df = data.DataReader(['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC'],'google', start, end)


# ** Creating a list of the ticker symbols (as strings) in alphabetical order. Call this list: tickers**

# In[5]:


tickers = ['BAC', 'C', 'GS', 'JPM', 'MS', 'WFC']


# ** Using pd.concat to concatenate the bank dataframes together to a single data frame called bank_stocks. Set the keys argument equal to the tickers list. Also pay attention to what axis you concatenate on.**

# In[6]:


bank_stocks = pd.concat([BAC, C, GS, JPM, MS, WFC],axis=1,keys=tickers)


# ** Set the column name levels (this is filled out for you):**

# In[7]:


bank_stocks.columns.names = ['Bank Ticker','Stock Info']


# ** Check the head of the bank_stocks dataframe.**

# In[8]:


bank_stocks.head()


# # EDA
# 
# Let's explore the data a bit! Before continuing
# 
# ** What is the max Close price for each bank's stock throughout the time period?**

# In[9]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').max()


# ** Creating a new empty DataFrame called returns. This dataframe will contain the returns for each bank's stock. returns are typically defined by:**
# 
# $$r_t = \frac{p_t - p_{t-1}}{p_{t-1}} = \frac{p_t}{p_{t-1}} - 1$$

# In[10]:


returns = pd.DataFrame()


# ** using pandas pct_change() method on the Close column to create a column representing this return value. I also Create a for loop that goes and for each Bank Stock Ticker creates this returns column and set's it as a column in the returns DataFrame.**

# In[11]:


for tick in tickers:
    returns[tick+' Return'] = bank_stocks[tick]['Close'].pct_change()
returns.head()


# ** Create a pairplot using seaborn of the returns dataframe. What stock stands out to you? Can you figure out why?**

# In[13]:


#returns[1:]
import seaborn as sns
sns.pairplot(returns[1:])


# Background on [Citigroup's Stock Crash available here.](https://en.wikipedia.org/wiki/Citigroup#November_2008.2C_Collapse_.26_US_Government_Intervention_.28part_of_the_Global_Financial_Crisis.29) 
# 
# You'll also see the enormous crash in value if you take a look a the stock price plot (which we do later in the visualizations.)

# ** Using this returns DataFrame, i can figure out on what dates each bank stock had the best and worst single day returns.4 of the banks share the same day for the worst drop, did anything significant happen that day?**

# In[ ]:


# Worst Drop (4 of them on Inauguration day)
returns.idxmin()


# [Citigroup had a stock split.](https://www.google.com/webhp?sourceid=chrome-instant&ion=1&espv=2&ie=UTF-8#q=citigroup+stock+2011+may)

# In[15]:


# Best Single Day Gain
# citigroup stock split in May 2011, but also JPM day after inauguration.
returns.idxmax()


# ** Taking  a look at the standard deviation of the returns, Citygroup is the riskiest**

# In[ ]:


returns.std() 


# In[17]:


returns.ix['2015-01-01':'2015-12-31'].std() 


# ** Create a distplot using seaborn of the 2015 returns for Morgan Stanley **

# In[18]:


sns.distplot(returns.ix['2015-01-01':'2015-12-31']['MS Return'],color='green',bins=100)


# ** Create a distplot using seaborn of the 2008 returns for CitiGroup **

# In[19]:


sns.distplot(returns.ix['2008-01-01':'2008-12-31']['C Return'],color='red',bins=100)


# ____
# # More Visualization
# 
# ### Imports

# In[20]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
get_ipython().magic('matplotlib inline')

# Optional Plotly Method Imports
import plotly
import cufflinks as cf
cf.go_offline()


# ** Creating a line plot showing Close price for each bank for the entire index of time.**

# In[21]:


for tick in tickers:
    bank_stocks[tick]['Close'].plot(figsize=(12,4),label=tick)
plt.legend()


# In[22]:


bank_stocks.xs(key='Close',axis=1,level='Stock Info').plot()


# In[23]:


# plotly
bank_stocks.xs(key='Close',axis=1,level='Stock Info').iplot()


# ## Moving Averages
# 
# Let's analyze the moving averages for these stocks in the year 2008. 
# 
# ** Plotting the rolling 30 day average against the Close Price for Bank Of America's stock for the year 2008**

# In[24]:


plt.figure(figsize=(12,6))
BAC['Close'].ix['2008-01-01':'2009-01-01'].rolling(window=30).mean().plot(label='30 Day Avg')
BAC['Close'].ix['2008-01-01':'2009-01-01'].plot(label='BAC CLOSE')
plt.legend()


# ** Creating a heatmap of the correlation between the stocks Close Price.**

# In[25]:


sns.heatmap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)


# ** Using seaborn's clustermap to cluster the correlations together:**

# In[26]:


sns.clustermap(bank_stocks.xs(key='Close',axis=1,level='Stock Info').corr(),annot=True)

