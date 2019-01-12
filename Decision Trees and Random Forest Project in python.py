
# coding: utf-8

# # Odlin Montila 
# 
# 
# 
# 
# 
# 
# 
# 
# # Random forest and Decision trees

# # Import Libraries
# 
# **Import the usual libraries for pandas and plotting. You can import sklearn later on.**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Getting  the Data
# 
# ** Using pandas to read loan_data.csv as a dataframe called loans.**

# In[2]:


loans = pd.read_csv('loan_data.csv')


# ** Checking out the info(), head(), and describe() methods on loans.**

# In[3]:


loans.info()


# In[4]:


loans.describe()


# In[5]:


loans.head()


# # Exploratory Data Analysis
# 
# 

# In[6]:


plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')


# ** Creating a similar figure, except this time i select by the not.fully.paid column.**

# In[7]:


plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')


# ** Creating a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid. **

# In[8]:


plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')


# ** Let's see the trend between FICO score and interest rate. Recreate the following jointplot.**

# In[9]:


sns.jointplot(x='fico',y='int.rate',data=loans,color='purple')


# ** Creating the following lmplots to see if the trend differed between not.fully.paid and credit.policy. Checking the documentation for lmplot() **

# In[10]:


plt.figure(figsize=(11,7))
sns.lmplot(y='int.rate',x='fico',data=loans,hue='credit.policy',
           col='not.fully.paid',palette='Set1')


# # Setting up the Data
# 
# 
# **Checking loans.info() again.**

# In[12]:


loans.info()


# ## Categorical Features
# 
# Notice that the **purpose** column as categorical
# 
# That means i need to transform them using dummy variables so sklearn will be able to understand them. I do this in one clean step using pd.get_dummies.
# 
# 
# 
# **Creating a list of 1 element containing the string 'purpose'. Call this list cat_feats.**

# In[36]:


cat_feats = ['purpose']


# **Now using pd.get_dummies(loans,columns=cat_feats,drop_first=True) to create a fixed larger dataframe that has new feature columns with dummy variables. Setting this dataframe as final_data.**

# In[37]:


final_data = pd.get_dummies(loans,columns=cat_feats,drop_first=True)


# In[38]:


final_data.info()


# ## Train Test Split
# 
# Now its time to split the data into a training set and a testing set!
# 
# ** Using sklearn to split the data into a training set and a testing set.**

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


# ## Training a Decision Tree Model
# 
# I start by training a single decision tree first
# 
# ** Import DecisionTreeClassifier**

# In[22]:


from sklearn.tree import DecisionTreeClassifier


# **Creating an instance of DecisionTreeClassifier() called dtree and fit it to the training data.**

# In[23]:


dtree = DecisionTreeClassifier()


# In[24]:


dtree.fit(X_train,y_train)


# ## Predictions and Evaluation of Decision Tree
# **Creating predictions from the test set and create a classification report and a confusion matrix.**

# In[25]:


predictions = dtree.predict(X_test)


# In[26]:


from sklearn.metrics import classification_report,confusion_matrix


# In[27]:


print(classification_report(y_test,predictions))


# In[28]:


print(confusion_matrix(y_test,predictions))


# ## Training the Random Forest model
# 
# Now its time to train the model!
# 
# **Creating an instance of the RandomForestClassifier class and fit it to the training data from the previous step.**

# In[29]:


from sklearn.ensemble import RandomForestClassifier


# In[30]:


rfc = RandomForestClassifier(n_estimators=600)


# In[31]:


rfc.fit(X_train,y_train)


# ## Predictions and Evaluation
# 
# Let's predict off the y_test values and evaluate our model.
# 
# ** Predict the class of not.fully.paid for the X_test data.**

# In[32]:


predictions = rfc.predict(X_test)


# **creating a classification report from the results.**

# In[33]:


from sklearn.metrics import classification_report,confusion_matrix


# In[34]:


print(classification_report(y_test,predictions))


# **Showing the Confusion Matrix for the predictions.**

# In[35]:


print(confusion_matrix(y_test,predictions))

