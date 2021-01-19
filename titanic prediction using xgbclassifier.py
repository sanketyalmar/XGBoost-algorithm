#!/usr/bin/env python
# coding: utf-8

# #Task: To build a predictive model that answers the question:
# “what sorts of people were more likely to survive? Database
# was provided of around nn customers along with their
# information.
# For this particular task, XGBoost algorithm was implemented
# considering the nature and expected outcome of the
# problem.
# The result successfully classified customers as per the
# requirement

# In[149]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[150]:


df=pd.read_csv("train.csv")


# In[151]:


df.head()


# In[152]:


df.isnull().sum()


# In[153]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False)


# In[154]:


df.shape


# In[155]:


df.info()


# In[156]:


df['Age']=df['Age'].fillna(df['Age'].mean())


# In[157]:


df.drop(['Cabin'],axis=1,inplace=True)


# In[158]:


df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])


# In[159]:


df.shape


# In[160]:


df.isnull().sum()


# In[161]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='coolwarm')


# In[162]:


df.head()


# In[163]:


columns=['Name','Sex','Ticket','Embarked']


# In[164]:


len('columns')


# In[165]:


def category_onehot_multcols(multcolumns):
    df_final=final_df
    i=0
    for fields in multcolumns:
        
        print(fields)
        df1=pd.get_dummies(final_df[fields],drop_first=True)
        
        final_df.drop([fields],axis=1,inplace=True)
        if i==0:
            df_final=df1.copy()
        else:
            
            df_final=pd.concat([df_final,df1],axis=1)
        i=i+1
       
        
    df_final=pd.concat([final_df,df_final],axis=1)
        
    return df_final


# In[166]:


main_df=df.copy()


# In[167]:


test_df=pd.read_csv("formulatedtest.csv")


# In[168]:


test_df.shape


# In[169]:


test_df.head()


# In[170]:


final_df=pd.concat([df,test_df],axis=0)


# In[171]:


final_df['Survived']


# In[172]:


final_df.shape


# In[173]:


final_df=category_onehot_multcols(columns)


# In[174]:


final_df =final_df.loc[:,~final_df.columns.duplicated()]


# In[175]:


final_df.shape


# In[176]:


final_df


# In[177]:


df_Train=final_df.iloc[:1422,:]
df_Test=final_df.iloc[1422:,:]


# In[178]:


df_Train.head()


# In[179]:


df_Test.head()


# In[180]:


df_Train.shape


# In[181]:


df_Test.drop(['Survived'],axis=1,inplace=True)


# In[182]:


X_train=df_Train.drop(['Survived'],axis=1)
y_train=df_Train['Survived']


# In[183]:


import xgboost
from xgboost import XGBClassifier


# In[184]:


classifier=xgboost.XGBClassifier()


# In[185]:


regressor=xgboost.XGBClassifier()


# In[186]:



booster=['gbtree','gblinear']
base_score=[0.25,0.5,0.75,1]


# In[187]:


n_estimators = [100, 500, 900, 1100, 1500]
max_depth = [2, 3, 5, 10, 15]
booster=['gbtree','gblinear']
learning_rate=[0.05,0.1,0.15,0.20]
min_child_weight=[1,2,3,4]

# Define the grid of hyperparameters to search
hyperparameter_grid = {
    'n_estimators': n_estimators,
    'max_depth':max_depth,
    'learning_rate':learning_rate,
    'min_child_weight':min_child_weight,
    'booster':booster,
    'base_score':base_score
    }


# In[188]:


from sklearn.model_selection import RandomizedSearchCV


# In[189]:


random_cv = RandomizedSearchCV(estimator=regressor,
            param_distributions=hyperparameter_grid,
            cv=5, n_iter=50,
            scoring = 'neg_mean_absolute_error',n_jobs = 4,
            verbose = 5, 
            return_train_score = True,
            random_state=42)


# In[191]:


X_train[:] = np.nan_to_num(X_train)
y_train[:] = np.nan_to_num(y_train)


# In[192]:


random_cv.fit(X_train,y_train)


# In[193]:


random_cv.best_estimator_


# In[194]:


random_cv.best_estimator_


# In[195]:


regressor=xgboost.XGBClassifier(base_score=0.25, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0,
       max_depth=2, min_child_weight=1, missing=None, n_estimators=900,
       n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=True, subsample=1)


# In[196]:


regressor.fit(X_train,y_train)


# In[197]:


import pickle
filename = 'finalized_model.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[199]:


df_Test.shape


# In[214]:


df_Test.head()


# In[220]:


y_pred=regressor.predict(df_Test)


# In[221]:


y_pred


# In[222]:


pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv('gender_submission.csv')
datasets=pd.concat([sub_df['PassengerId'],pred],axis=1)
datasets.columns=['PassengerId','Survived']
datasets.to_csv('gender_submission.csv',index=False)


# In[208]:


pwd


# In[ ]:




