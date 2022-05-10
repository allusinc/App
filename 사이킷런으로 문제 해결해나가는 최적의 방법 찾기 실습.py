#!/usr/bin/env python
# coding: utf-8

# In[6]:


from sklearn.datasets import load_iris


# In[7]:


data=load_iris()


# In[8]:


import pandas as pd


# In[9]:


data.data


# In[10]:


print(data.DESCR)


# In[12]:


data


# In[11]:


data.feature_names


# In[12]:


feature=pd.DataFrame(data.data, columns=data.feature_names)


# In[13]:


target=pd.DataFrame(data.target, columns=['species'])


# In[14]:


rename.feature.columns=['sepal_length','sepal_width','petal_length','petal_width']


# In[15]:


feature=pd.DataFrame(data.data, columns=['s_l','s_w','p_l','p_w'])


# In[16]:


target=pd.DataFrame(data.target, columns=['species'])


# In[17]:


iris=pd.concat([feature,target],axis=1)


# In[35]:


#iris.rename({'sepal length':'s_l'...}, axis=1, inplace=True)


# In[18]:


iris.head()


# In[19]:


data.target_names #lambda 함수 이용해서 변경


# In[20]:


data.target_names[2]


# In[21]:


iris['species']=iris.species.map(lambda x: data.target_names[x])


# In[22]:


iris.head()


# In[23]:


iris.isna().sum()


# In[24]:


iris.info()


# In[25]:


iris.describe()


# In[51]:


iris.corr()


# 상관관계가 높은 피처는 다중공선성 문제를 유발할수 있으므로 둘 중 하나의 변수만 선택해서 사용하는 것이 좋을 것으로 보인다.

# In[26]:


iris.groupby('species').size()


# Iris 데이터에 대한 전처리 및  EDA

# In[27]:


from sklearn.model_selection import train_test_split


# In[28]:


iris.head()


# In[29]:


X_train, X_test, y_train, y_test=train_test_split(data.data, data.target, test_size=0.33, random_state=42)


# In[30]:


from sklearn.tree import DecisionTreeClassifier


# In[31]:


data.target


# In[32]:


model=DecisionTreeClassifier()


# #help(model)

# In[33]:


model.fit(X_train,y_train)


# In[34]:


model.score(X_test,y_test)


# In[27]:


## cross_validation


# In[35]:


from sklearn.model_selection import cross_val_score, KFold


# In[36]:


cv=KFold(n_splits=10,shuffle=True, random_state=42)


# In[37]:


results=cross_val_score(model, X_train, y_train, cv=cv)


# In[38]:


results


# In[39]:


for i, _ in enumerate(results):
    print("{}th cross calidation score {}".format(i,_))


# In[40]:


import numpy as np


# In[41]:


np.mean(results)


# In[42]:


get_ipython().system('pip install scikit-plot')


# In[43]:


import scikitplot as skplt


# In[44]:


import matplotlib.pyplot as plt


# In[45]:


skplt.estimators.plot_learning_curve(model, X_train,y_train,figsize=(6,6))
plt.show()


# In[83]:


# 최적의 모델 선택을 위한 하이퍼파라미터 찾기


# In[46]:


estimator=DecisionTreeClassifier()


# In[47]:


from sklearn.model_selection import GridSearchCV


# In[49]:


parameters={"max_depth":[4,6,8,10,12],'criterion':['gini','entropy'],'splitter':['best','random'],
           'min_weight_fraction_leaf':[0.0,0.1,0.2,0.3], 'random_state':[7,23,42,78,142],'min_impurity_decrease':[0.0,0.05,0.1,0.2]}


# In[51]:


model2=GridSearchCV(estimator=estimator,param_grid=parameters,cv=KFold(10),verbose=1,n_jobs=-1,refit=True)


# In[52]:


model2.fit(X_train, y_train)


# In[53]:


model2.best_estimator_


# In[54]:


model2.best_params_


# In[55]:


model2.best_score_


# In[56]:


from sklearn.metrics import accuracy_score


# In[57]:


model2.predict(X_test)


# ![%EB%B6%84%EB%A5%98%EB%AA%A9%EC%A0%81%20%EB%AA%A8%EB%8D%B8%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9C.JPG](attachment:%EB%B6%84%EB%A5%98%EB%AA%A9%EC%A0%81%20%EB%AA%A8%EB%8D%B8%ED%8F%89%EA%B0%80%EC%A7%80%ED%91%9C.JPG)

# In[58]:


from sklearn.metrics import confusion_matrix


# In[59]:


pred=model2.predict(X_test)


# In[61]:


confusion_matrix(y_test,pred)


# In[80]:


(19+15+15)/(19+15+15+1)


# In[81]:


import scikitplot as skplt


# In[82]:


skplt.metrics.plot_confusion_matrix(y_test,pred, figsize=(8,6))
plt.show()


# In[63]:


from sklearn.metrics import precision_score


# In[64]:


precisions=precision_score(y_test,pred,average=None)


# In[65]:


precisions


# In[66]:


data.target_names


# In[67]:


for target, score in zip(data.target_names, precisions):
    print(f"{target} precision:{score}")


# In[68]:


from sklearn.metrics import recall_score


# In[70]:


recalls=recall_score(y_test,pred,average=None)
for target, score in zip(data.target_names, recalls):
    print(f"{target} precision:{score}")


# In[71]:


from sklearn.metrics import fbeta_score, f1_score


# In[73]:


fbetas=fbeta_score(y_test,pred,beta=1,average=None)


# In[74]:


for target, score in zip(data.target_names, fbetas):
    print(f"{target} precision:{score}")


# In[75]:


f1s=f1_score(y_test,pred,average=None)


# In[76]:


for target, score in zip(data.target_names, f1s):
    print(f"{target} precision:{score}")


# In[77]:


from sklearn.metrics import classification_report


# In[78]:


print(classification_report(y_test,pred))


# In[79]:


#ROC 커브


# In[ ]:




