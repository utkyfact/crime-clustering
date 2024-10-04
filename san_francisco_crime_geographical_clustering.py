#!/usr/bin/env python
# coding: utf-8

# # SAN FRANCISCO CRIME GEOGRAPHICAL CLUSTERING PROJECT



# In[ ]:


from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("train.csv")
df.head()


# In[ ]:



df = df.drop(['PdDistrict', 'Address', 'Resolution', 'Descript', 'DayOfWeek'], axis = 1) # axis = 1 for column drop, 0 for row drop..


# In[ ]:


df.tail(5)


# In[ ]:


df.isnull().sum()



# In[ ]:



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


df_2014 = df[(df.Dates == '2014')]
df_2014.head()


# In[ ]:


scaler = MinMaxScaler()


scaler.fit(df_2014[['X']])
df_2014['X_scaled'] = scaler.transform(df_2014[['X']]) 

scaler.fit(df_2014[['Y']])
df_2014['Y_scaled'] = scaler.transform(df_2014[['Y']])



# In[ ]:


df_2014.head()




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



# In[ ]:


model = KMeans(n_clusters=5)
y_predicted = model.fit_predict(df_2014[['X_scaled','Y_scaled']])
y_predicted


# In[ ]:


df_2014['cluster'] = y_predicted
df_2014



# In[ ]:



import plotly.express as px


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




