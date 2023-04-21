#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # Data

# In[2]:


data=pd.read_csv(r"C:\Users\Parimal\Downloads\keywords (1).csv")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data.isnull().sum()


# In[6]:


data.replace(np.nan,0)


# In[7]:


data.describe()


# In[8]:


data['Type'].value_counts()


# In[9]:


data['Subtype'].value_counts()


# In[10]:


data['Keyword'].value_counts()


# In[11]:


data['Keyword'].nunique()  ## Warning - Your data is of 80observation which is quite and also 73 keyward are unique out of 80 which cannot make a good pattern for model


# # Label Encoding

# In[12]:


from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()


# In[13]:


data['Keyword']=le.fit_transform(data['Keyword'])
data['Type']=le.fit_transform(data['Type'])
data['Subtype']=le.fit_transform(data['Subtype'])


# In[14]:


data


# # Data slicing

# In[15]:


x=data.iloc[:,1:]
y=data.iloc[:,0]
print(type(x),"\n",type(y))


# In[ ]:





# # Imbalancing

# In[16]:


from imblearn.over_sampling import SMOTE                 # Create artificial variable in minority counts
sm = SMOTE()
x_data, y_data = sm.fit_resample(x,y)


# # Feature scalling

# In[17]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_data=scaler.fit_transform(x_data)
#data=pd.DataFrame(data,columns=['Type',"Subtype","Keyword"])


# In[18]:


#print(x_data,y_data)


# # Train test

# In[19]:


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.20,random_state=1)


# # model

# In[20]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)


# In[21]:


y_pred=model.predict(x_test)


# # Accuracy

# In[22]:


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy


# In[23]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test,y_pred)


# In[ ]:




