#!/usr/bin/env python
# coding: utf-8

# <b>
# Presented by Vinit Yadav
#     
# The Sparks Foundation : Data Science and Business Analytics Intern
#     
# Task 2- Unsupervised Machine Learning 
#     
# Objective : For the given 'Iris' dataset, predict the optimum number of clusters and represent it visually.
#     
#     
#     
#     
# Importing libraries 
# </b>

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import datasets


# In[2]:


iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names)


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.corr()


# In[6]:


sns.heatmap(df.corr(),annot=True,cmap='viridis')


# In[7]:


x=df.values


# In[8]:


x


# In[9]:


from sklearn.cluster import KMeans


# In[10]:


wcss=[]  # Within cluster sum of squares


# In[11]:


for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)


# In[12]:


plt.title('The Elbow Method')
plt.plot(range(1,11),wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()


# In[13]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)


# In[14]:


plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')
plt.legend()

