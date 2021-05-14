#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import pandas as pd
from flask import Flask,render_template,request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# In[18]:


data=pd.read_csv('movie_data.csv')


# In[20]:


data['comb'] = data['actor_1_name'] + ' ' + data['actor_2_name'] + ' '+ data['actor_3_name'] + ' '+ data['director_name'] +' ' + data['genres']


# In[23]:


cv=CountVectorizer()
count_metrix=cv.fit_transform(data['comb'])
sim=cosine_similarity(count_metrix)


# In[26]:


np.save('simliarity_matrix',sim)
data.to_csv('movie_data.csv',index=False)

