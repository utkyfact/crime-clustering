#!/usr/bin/env python
# coding: utf-8

# # SAN FRANCISCO CRIME GEOGRAPHICAL CLUSTERING PROJECT
# 
# In this lecture, we will perform geographic clustering using the San Francisco crime dataset on Kaggle. You can download the dataset from Kaggle: https://www.kaggle.com/c/sf-crime/data
# Please download only train.csv.zip file and unzip the file to the same directory with this Python source code..
# 
# The operation steps for this project is as follows:
# 
# 1. Import libraries and prepare the dataset.
# 2. Decide how many clusters we will have using Elbow Method (We will find K value).
# 3. Build the model and perform the clustering operation using K-Means Machine Learning algorithm.
# 4. Visualize our clustering results on well known geographic map system (OpenStreetMap)
# 5. Finally we will export our resulting geographic map into a html file so that it can be in any web site easily.
# 
# For all the above tasks including geographic map drawing we will use only Python ! Before starting to our project you should install plotly Python library to your Anaconda 3 environment. # You can install plotly to your Anaconda Environment using the following command from Anaconda prompt:
# 
# conda install plotly
# 

# ## STEP 1. Import libraries and prepare the dataset

# In[ ]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# You can download the dataset from Kaggle: https://www.kaggle.com/c/sf-crime/data
# Please download only train.csv.zip file and unzip the file to the same directory with this Python source code..
df = pd.read_csv("train.csv")
df.head()


# In[ ]:


# Dropping unnecessary columns PdDistrict, PdDistrict since we will use only Lat, Lng for Clustering..
# Also dropping Resolution, Descript,.. for we dont need them for clustering 

df = df.drop(['PdDistrict', 'Address', 'Resolution', 'Descript', 'DayOfWeek'], axis = 1) # axis = 1 for column drop, 0 for row drop..


# In[ ]:


df.tail(5)


# In[ ]:


# We have no null data see:
df.isnull().sum()


# ### This dataset contains information from 2003 to 2015. We will use only data in year 2014. 
# 
# For this purpose we will perform year filtering operation now..

# In[ ]:


# For year filtering operation..

f = lambda x: (x["Dates"].split())[0] 
df["Dates"] = df.apply(f, axis=1)
df.head()


# In[ ]:


f = lambda x: (x["Dates"].split('-'))[0] 
df["Dates"] = df.apply(f, axis=1)
df.head()


# In[ ]:


df.tail()


# In[ ]:


# Categorize dataset by year otherwise too long to process.. 
# We will use only year 2014 values for this project, but you can change this easily if you want..
df_2014 = df[(df.Dates == '2014')]
df_2014.head()


# In[ ]:


# We scale the data for accurate results...
scaler = MinMaxScaler()

# Y is latitude and X is longitude... 
# Any location in Earth can be described using Latitude and Longitude geographics coordinate values.

scaler.fit(df_2014[['X']])
df_2014['X_scaled'] = scaler.transform(df_2014[['X']]) 

scaler.fit(df_2014[['Y']])
df_2014['Y_scaled'] = scaler.transform(df_2014[['Y']])

# Please notice we have stored scaled values in new columns (X_scaled and Y_scaled), since we will use original values 
# in geographic operations later..


# In[ ]:


df_2014.head()


# ## STEP 2. Decide how many clusters we will have using Elbow Method (We will find K value).
# 
# K is a hyper-parameter (designer must have decide the value of K). Here K is the number of clusters, we tell the model how many clusters we want. 
# 
# But how can we decide the value of K ? Don't worry, there is a method called Elbow Method for defining hyper parameter K...
# 
# For this purpose we will use The Elbow Method, try K values from 1 to 15, and find the best K value.
# 

# In[ ]:


k_range = range(1,15)

list_dist = []

for k in k_range:
    model = KMeans(n_clusters=k)
    model.fit(df_2014[['X_scaled','Y_scaled']])
    list_dist.append(model.inertia_)


# In[ ]:


from matplotlib import pyplot as plt

plt.xlabel('K')
plt.ylabel('Distortion value (inertia)')
plt.plot(k_range,list_dist)
plt.show()


# #### Using Elbow method I decided to use K = 5 in my model...

# ## STEP 3. Build the model and perform the clustering operation using K-Means Machine Learning algorithm

# In[ ]:


# Let's build a K-Means model for K = 5:
model = KMeans(n_clusters=5)
y_predicted = model.fit_predict(df_2014[['X_scaled','Y_scaled']])
y_predicted


# In[ ]:


df_2014['cluster'] = y_predicted
df_2014


# ## STEP 4. Visualize our clustering results 
# ## Geographical Map Building using our Machine Learning model results...

# In[ ]:


# For Geographical Map Drawing we will use plotly library. 
# You can install plotly to your Anaconda Environment using the following command from Anaconda prompt:
# conda install plotly

import plotly.express as px   # You can install plotly module with: "conda install plotly" in Anaconda prompt.. 


# In[ ]:


# Don't forget Y is latiutude and X is longitude...
figure = px.scatter_mapbox(df_2014, lat='Y', lon='X',                       
                       center = dict(lat = 37.8, lon = -122.4), # This is the coordinate of San Francisco..
                       zoom = 9,                                # Zoom of the map
                       opacity = .9,                           # opacity of the map a value between 0 and 1..
                       mapbox_style = 'stamen-terrain',       # basemap 
                       color = 'cluster',                      # Map will draw scatter colors according to cluster number..
                       title = 'San Francisco Crime Districts',
                       width = 1100,
                       height = 700,                     
                       hover_data = ['cluster', 'Category', 'Y', 'X']
                       )

figure.show()


# ## STEP 5. Finally we will export our resulting geographic map into a html file so that it can be used in any web site easily

# In[ ]:


import plotly
plotly.offline.plot(figure, filename = 'maptest.html', auto_open = True)


# In[ ]:


# if you want to use another basemap or use other methods of plotly you can get info using help(px.scatter_mapbox):
help(px.scatter_mapbox)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




