#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[5]:


dataset=pd.read_csv('Churn_Modelling.csv')


# In[6]:


dataset.head()


# In[7]:


dataset.info()


# In[8]:


dataset.describe()


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[10]:


dataset['CreditScore'].hist()


# In[ ]:


import numpy as np


# In[13]:


np.info()


# In[14]:


dataset


# In[15]:


dataset1=pd.get_dummies(dataset,columns=['Gender','Geography'])


# In[16]:


dataset.head(5)


# In[17]:


dataset1.head(5)


# In[ ]:





# In[3]:





# In[18]:


x=dataset1.drop(['Exited','RowNumber','CustomerId','Surname'], axis=1)
y=dataset1['Exited']


# In[19]:


x.head(5)


# In[20]:


y.head(5)


# In[21]:


x.tail(5)


# In[22]:


from sklearn.model_selection import train_test_split


# In[28]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.25)


# In[29]:


from sklearn.tree import DecisionTreeClassifier


# In[31]:


classifier = DecisionTreeClassifier()


# In[ ]:





# In[32]:


classifier.fit(x_train,y_train)


# In[33]:


pred = classifier.predict(x_test)


# In[34]:


from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,pred)


# In[35]:


print(accuracy)


# In[36]:


yhat=classifier.predict(x)


# In[37]:


dataset['yhat']=yhat


# In[38]:


accuracy=accuracy_score(y,yhat)


# In[39]:


print(accuracy)


# In[40]:


pd.DataFrame(dataset).to_csv("Bank Churn_with_predictions.csv")

