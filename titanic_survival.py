#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df=pd.read_csv("Titanic-Dataset.csv")


# In[3]:


df.head()
df


# In[4]:


df.isnull().sum()


# In[5]:


x=df[["Pclass","Sex","Age"]]
y=df.Survived


# In[6]:


x.isnull().sum()


# In[7]:


x.Age.fillna(x.Age.mean(),inplace=True)


# In[8]:


x


# converting string to numerical

# In[9]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
gen=le.fit_transform(x.Sex)


# In[10]:


x["Gender"]=gen
x.drop("Sex",axis=1,inplace=True)


# In[11]:


x.isnull().sum()


# In[12]:


y


# In[13]:


from sklearn.model_selection import train_test_split


# In[14]:


train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.2)


# In[15]:


len(train_x)


# In[16]:


len(test_x)


# In[17]:


from sklearn.linear_model import LogisticRegression


# In[18]:


model=LogisticRegression()


# In[19]:


model.fit(train_x,train_y)


# In[20]:


model.score(test_x,test_y)


# In[21]:


x.head()


# In[22]:


if model.predict([[3,70,1]])[0]==1:
    print("survived")
else:
    print("not survived")


# In[23]:


df.head()
df


# In[24]:


titanic = df


# In[25]:


titanic.set_index('PassengerId', inplace=True)


# In[26]:


titanic['isMale'] = (titanic.Sex == 'male') * 1


# In[27]:


titanic.isMale.plot(kind='pie')


# In[28]:


import matplotlib.pyplot as plt


# In[29]:


titanic.Sex.value_counts()


# In[30]:


titanic.Sex.value_counts().plot(kind='pie')


# scatterplot with fare paid and the age, differ the plot color by gender

# In[31]:


import seaborn as sns


# In[32]:


sns.scatterplot(x='Age', y='Fare', hue='Sex', data=titanic)


# In[33]:


titanic.Survived.sum()


# histogram with fare paid

# In[34]:


sns.distplot(titanic.Fare)


# In[36]:


import pickle
with open('titanic_model.pkl', 'wb') as file:
    pickle.dump(model, file)


# In[37]:


pip install flask


# In[ ]:





# In[ ]:





# In[ ]:




