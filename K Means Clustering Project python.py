
# coding: utf-8

# # Odlin Montila

# # K Means Clustering Project -
# 
# For this project i will attempt to use KMeans Clustering to cluster Universities into to two groups, Private and Public.

# ## Importing Libraries
# 
# 

# In[103]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')


# ## Getting the Data

# ** Reading in the College_Data file using read_csv.**

# In[104]:


df = pd.read_csv('College_Data',index_col=0)


# **Checking the head of the data**

# In[105]:


df.head()


# ** Checking the info() and describe() methods on the data.**

# In[106]:


df.info()


# In[107]:


df.describe()


# ## EDA
# 
# It's time to create some data visualizations!
# 
# ** Creating a scatterplot of Grad.Rate versus Room.Board where the points are colored by the Private column. **

# In[111]:


sns.set_style('whitegrid')
sns.lmplot('Room.Board','Grad.Rate',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)


# **Creating a scatterplot of F.Undergrad versus Outstate where the points are colored by the Private column.**

# In[112]:


sns.set_style('whitegrid')
sns.lmplot('Outstate','F.Undergrad',data=df, hue='Private',
           palette='coolwarm',size=6,aspect=1,fit_reg=False)


# ** Creating a stacked histogram showing Out of State Tuition based on the Private column.**

# In[109]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# **Creating a similar histogram for the Grad.Rate column.**

# In[110]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# ** Notice how there seems to be a private school with a graduation rate of higher than 100%.I am going to find the name of that school?**

# In[113]:


df[df['Grad.Rate'] > 100]


# ** Set that school's graduation rate to 100 so it makes sense.**

# In[93]:


df['Grad.Rate']['Cazenovia College'] = 100


# In[94]:


df[df['Grad.Rate'] > 100]


# In[95]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# ## K Means Cluster Creation
# 
# Now it is time to create the Cluster labels!
# 
# ** Import KMeans from SciKit Learn.**

# In[114]:


from sklearn.cluster import KMeans


# ** Creating an instance of a K Means model with 2 clusters.**

# In[115]:


kmeans = KMeans(n_clusters=2)


# **Fit the model to all the data except for the Private label.**

# In[116]:


kmeans.fit(df.drop('Private',axis=1))


# ** What are the cluster center vectors?**

# In[117]:


kmeans.cluster_centers_


# ## Evaluation
# 
# 
# ** Creating a new column for df called 'Cluster', which is a 1 for a Private school, and a 0 for a public school.**

# In[118]:


def converter(cluster):
    if cluster=='Yes':
        return 1
    else:
        return 0


# In[119]:


df['Cluster'] = df['Private'].apply(converter)


# In[122]:


df.head()


# ** Creating a confusion matrix and classification report to see how well the Kmeans clustering worked without being given any labels.**

# In[123]:


from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))

