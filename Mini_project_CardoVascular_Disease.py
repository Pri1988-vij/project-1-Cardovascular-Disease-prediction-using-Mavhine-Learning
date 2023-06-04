#!/usr/bin/env python
# coding: utf-8

# # Perform data pre-processing operations.

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv('cardo_train.csv',sep=";")
df


# In[3]:


df.describe()


# In[4]:


# df.dropna()


# In[5]:


df.columns


# In[6]:


df.isnull().values.any()


# In[7]:


df.head()


# In[8]:


df.tail()


# # data analysis and visualizations draw all the possible plots to provide essential informations and to derive some meaningful insights.

# In[9]:


import matplotlib.pyplot as plt
x=df['cholesterol']
y=df['cardio']
plt.plot(x,y,color='r')
plt.xlabel('cholesterol')
plt.ylabel('cardio')
plt.title("clolestrol vs cardio")
plt.show()


# In[10]:


x=df['smoke']
y=df['cardio']
plt.plot(x,y,color='r')
plt.xlabel('smoke')
plt.ylabel('cardio')
plt.title("smoke vs cardio")
plt.show()


# In[11]:



x=df['alco']
y=df['cardio']
plt.scatter(x, y, color='blue', marker='*')
plt.xlabel('alco')
plt.ylabel('cardio')
plt.title("alco vs cardio")
plt.show()


# In[12]:



import seaborn as sns
sns.lineplot(x="active",y="cardio",data = df)
plt.xlabel('active')
plt.ylabel('cardio')
plt.title("active vs cardio")
plt.show()


# In[13]:


import seaborn as sns
sns.histplot(x="gluc",y="cardio",data = df)
plt.xlabel('gluc')
plt.ylabel('cardio')
plt.title("gluc vs cardio")
plt.show()


# In[14]:


import seaborn as sns
sns.lmplot(x="ap_hi",y="cardio",data = df)
plt.xlabel('ap_hi')
plt.ylabel('cardio')
plt.title("ap_hi vs cardio")
plt.show()


# In[15]:


import seaborn as sns
sns.scatterplot(x="ap_lo",y="cardio",data = df)
plt.xlabel('ap_lo')
plt.ylabel('cardio')
plt.title("ap_lo vs cardio")
plt.show()


# In[16]:


import seaborn as sns
sns.barplot(x="active",y="cardio",data = df)
plt.xlabel('active')
plt.ylabel('cardio')
plt.title("active vs cardio")
plt.show()


# # Show your correlation matrix of features according to the datasets.

# In[17]:


import pandas as pd
import seaborn as sns

data = pd.read_csv('cardo_train.csv',sep=";")

corr_matrix = data.corr()

sns.heatmap(corr_matrix, annot=True)


# In[18]:


x=df.iloc[:,0:12]
x


# In[19]:


y=df.iloc[:,12:]
y


# In[20]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.7)


# In[21]:


x_train


# In[22]:


y_train


# In[23]:


x_test


# In[24]:


y_test


# # Linear Regression

# In[25]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[26]:


model.fit(x_train,y_train)


# In[27]:


model.score(x_test,y_test)


# In[28]:


predicted=model.predict(x_test)
predicted


# In[29]:


model.coef_


# In[30]:


model.intercept_


# # Logistic Regression

# In[1]:


from sklearn.linear_model import LogisticRegression


# In[32]:


model2= LogisticRegression()


# In[33]:


model2.fit(x_train,y_train)


# In[34]:


model2.score(x_test,y_test)


# In[35]:


predicted=model2.predict(x_test)
predicted


# In[36]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,predicted)
cm


# In[37]:


import seaborn as sns
plt.figure(figsize=(6,6))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("actual")
plt.show()


# # Random forest

# In[38]:


# random forest
from sklearn.ensemble import RandomForestRegressor
model3 = RandomForestRegressor()


# In[39]:


model3.fit(x_train,y_train)


# In[40]:


model3.score(x_test,y_test)


# In[41]:


pred=model3.predict(x_test)
pred


# # Decision tree

# In[42]:


# decision tree

from sklearn.tree import DecisionTreeClassifier
model4=DecisionTreeClassifier(criterion="entropy")
model4.fit(x_train,y_train)


# In[43]:


model4.score(x_test,y_test)


# In[44]:


pred=model4.predict(x_test)
pred


# In[45]:


# from sklearn import tree
# tree.plot_tree(model4)


# # k nearest Neighbors

# In[46]:


#knn classifier

from sklearn.neighbors import KNeighborsClassifier
model5=KNeighborsClassifier()
model5.fit(x_train,y_train)


# In[47]:


model5.score(x_test,y_test)


# In[48]:


pred=model.predict(x_test)
pred


# In[49]:


plt.scatter(df.active,df.cardio)


# # K Means Clustering

# In[57]:


from sklearn.cluster import KMeans


# In[58]:


df


# In[60]:


km=KMeans(n_clusters=5)
km.fit(df[["id","age","gender","height","weight","ap_hi","ap_lo","cholesterol","gluc","smoke","alco","active","cardio"]])


# In[61]:


km.cluster_centers_


# In[62]:


df["cluster_group"]=km.labels_


# In[63]:


df


# In[64]:


df["cluster_group"].value_counts()


# In[74]:


sns.scatterplot(x="gender",y="cardio",data=df,hue="cluster_group")


# # support vector machine

# In[50]:


#svm

from sklearn.svm import SVC
model6 = SVC()


# In[54]:


x=df.iloc[:,0:12]
x


# In[56]:


# import numpy as np

# # Create a column vector y
# y = df.iloc[:,12:0]

# # Convert y to a 1D array using ravel()
# y_1d = np.ravel(y)

# # Check the shape of y_1d
# print(y_1d.shape)  


# In[52]:


model6.fit(x_train,y_train)


# In[53]:


model6.score(x_test,y_test)


# In[55]:


pred=model6.predict(x_test)
pred


# In[ ]:




