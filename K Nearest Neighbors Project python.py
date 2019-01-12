
# coding: utf-8

# # Odlin Montila

# # K Nearest Neighbors Project 
# 
# 
# ## Import Libraries
# **Importing pandas,seaborn, and the usual libraries.**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Getting the Data
# ** Reading the 'KNN_Project_Data csv file into a dataframe **

# In[2]:


df = pd.read_csv('KNN_Project_Data')


# **Checking the head of the dataframe.**

# In[23]:


df.head() 


# # EDA
# 
# Since this data is artificial, i'll just do a large pairplot with seaborn.
# 
# **Using seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.**

# In[4]:


# THIS IS GOING TO BE A VERY LARGE PLOT
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')


# # Standardize the Variables
# 
# Time to standardize the variables.
# 
# ** Import StandardScaler from Scikit learn.**

# In[5]:


from sklearn.preprocessing import StandardScaler


# ** Creating a StandardScaler() object called scaler.**

# In[6]:


scaler = StandardScaler()


# ** Fitting scaler to the features.**

# In[7]:


scaler.fit(df.drop('TARGET CLASS',axis=1))


# **Using the .transform() method to transform the features to a scaled version.**

# In[8]:


scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# **Converting the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[9]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# # Train Test Split
# 
# **Using train_test_split to split the data into a training set and a testing set.**

# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['TARGET CLASS'],
                                                    test_size=0.30)


# # Using KNN
# 
# **Importing KNeighborsClassifier from scikit learn.**

# In[12]:


from sklearn.neighbors import KNeighborsClassifier


# **Creating a KNN model instance with n_neighbors=1**

# In[13]:


knn = KNeighborsClassifier(n_neighbors=1)


# **Fitting this KNN model to the training data.**

# In[14]:


knn.fit(X_train,y_train)


# # Predictions and Evaluations
# Let's evaluate the KNN model!

# **Using the predict method to predict values using your KNN model and X_test.**

# In[24]:


pred = knn.predict(X_test)


# ** Creating a confusion matrix and classification report.**

# In[16]:


from sklearn.metrics import classification_report,confusion_matrix


# In[17]:


print(confusion_matrix(y_test,pred))


# In[18]:


print(classification_report(y_test,pred))


# # Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# ** Creating a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list.**

# In[25]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# **Now create the following plot using the information from your for loop.**

# In[20]:


plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ## Retrain with new K Value
# 
# **Retrainning the model with the best K value,and re-do the classification report and the confusion matrix.**

# In[21]:


# NOW WITH K=30
knn = KNeighborsClassifier(n_neighbors=30)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=30')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))

