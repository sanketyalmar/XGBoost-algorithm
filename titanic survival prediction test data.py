#!/usr/bin/env python
# coding: utf-8

# In[52]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[53]:


pwd


# In[54]:


test_df=pd.read_csv("test.csv")


# In[55]:


test_df.shape


# In[56]:


test_df.head()


# In[57]:


test_df.isnull().sum()


# In[58]:


test_df['Age']=test_df['Age'].fillna(test_df['Age'].mean())


# In[59]:


test_df['Fare']=test_df['Fare'].fillna(test_df['Fare'].mean())


# In[60]:


test_df.drop(['Cabin'],axis=1,inplace=True)


# In[61]:


test_df.shape


# In[63]:


sns.heatmap(test_df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[64]:


test_df.loc[:, test_df.isnull().any()].head()


# In[65]:


test_df.shape


# In[66]:


test_df.to_csv('formulatedtest.csv',index=False)


# In[ ]:




