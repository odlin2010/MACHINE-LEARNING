
# coding: utf-8

# 
# # Odlin Montila
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# ** Using pandas to read in the bank_note_data.csv file **

# In[1]:


import pandas as pd


# In[3]:


data = pd.read_csv('bank_note_data.csv')


# ** Checking the head of the Data **

# In[61]:


data.head()


# 
# 
# ** Importing seaborn and set matplolib inline for viewing **

# In[67]:


import seaborn as sns
get_ipython().magic('matplotlib inline')


# ** Creating a Countplot of the Classes (Authentic 1 vs Fake 0) **

# In[68]:


sns.countplot(x='Class',data=data)


# ** Creating a PairPlot of the Data with Seaborn, setting Hue to Class **

# In[69]:


sns.pairplot(data,hue='Class')


# ## Data Preparation 
# 
# 
# 
# ### Standard Scaling
# 
# ** 

# In[71]:


from sklearn.preprocessing import StandardScaler


# **Creating a StandardScaler() object called scaler.**

# In[72]:


scaler = StandardScaler()


# **Fitting scaler to the features.**

# In[73]:


scaler.fit(data.drop('Class',axis=1))


# **Using the .transform() method to transform the features to a scaled version.**

# In[74]:


scaled_features = scaler.fit_transform(data.drop('Class',axis=1))


# **Converting the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.**

# In[77]:


df_feat = pd.DataFrame(scaled_features,columns=data.columns[:-1])
df_feat.head()


# ## Train Test Split
# 
# ** Creating two objects X and y which are the scaled feature values and labels respectively.**

# In[79]:


X = df_feat


# In[80]:


y = data['Class']


# In[81]:


X = X.as_matrix()
y = y.as_matrix()


# ** Using SciKit Learn to create training and testing sets of the data **

# In[45]:


from sklearn.cross_validation import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# # Contrib.learn
# 
# ** Import tensorflow.contrib.learn.python.learn as learn**

# In[82]:


import tensorflow.contrib.learn.python.learn as learn


# ** Creating an object called classifier which is a DNNClassifier from learn. Set it to have 2 classes and a [10,20,10] hidden unit layer structure:**

# In[83]:


classifier = learn.DNNClassifier(hidden_units=[10, 20, 10], n_classes=2)


# ** Now fit classifier to the training data. Use steps=200 with a batch_size of 20.**
# 
# *Note:  I ignore any warnings i get, they won't effect the  output*

# In[94]:


classifier.fit(X_train, y_train, steps=200, batch_size=20)


# ## Model Evaluation
# 
# ** Using the predict method from the classifier model to create predictions from X_test **

# In[95]:


note_predictions = classifier.predict(X_test)


# ** Now i create a classification report and a Confusion Matrix.**

# In[96]:


from sklearn.metrics import classification_report,confusion_matrix


# In[97]:


print(confusion_matrix(y_test,note_predictions))


# In[98]:


print(classification_report(y_test,note_predictions))

