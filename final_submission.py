#!/usr/bin/env python
# coding: utf-8

# ### importing required libraries 

# In[1]:



import pandas as pd
import math
import numpy as np
from category_encoders import *
from catboost import CatBoostRegressor
from sklearn import ensemble, preprocessing, model_selection, metrics
from sklearn.preprocessing import StandardScaler


# ### function for data cleaning and handling categorical data 

# In[2]:



def cleaning(df):
    df.rename(columns={'Instance':'instance','Year of Record':'record_year','Gender':'gender','Age':'age',
                       'Country':'country','Size of City':'city_size','Profession':'profession',
                       'University Degree':'degree','Wears Glasses':'glasses','Hair Color':'hair_color',
                       'Body Height [cm]':'height','Income in EUR':'income','Income':'income'},inplace=True)
    
#     converting to lowercase
    for col in df:
        if col in ['gender', 'country', 'profession', 'degree']:
            df[col] = df[col].str.lower()

#     remove columns glasses and hair color, as deemed un-correlated from heatmaps
    df = df.drop(['instance', 'glasses', 'hair_color'], axis=1)

    
#     For profession, identify major professions

    df['is_analyst']=np.where(df['profession'].str.contains('analyst'),1,0)
    df['is_manager']=np.where(df['profession'].str.contains('manager'),1,0)
    df['is_senior']=np.where(df['profession'].str.contains('senior'),1,0)

#     and take first 5 characters from all, based on trial and errror with correlation
    df.profession = df.profession.str[:5]
    
#     filling median year in blank record_year
    df.record_year = df.record_year.fillna(df.record_year.median())
    
#     median age for null age
    df.age = df.age.fillna(df.age.median())


#     using one hot encoding in column gender
    df.gender.replace([0, np.nan], '0', inplace=True)
    df.gender = map(lambda x: x.lower(), df.gender)
    df_gender = pd.get_dummies(df.gender)
    df = pd.concat([df, df_gender], axis=1)
    df = df.drop('gender', axis=1)
       
    
#     labelencoding degree
    df.degree.replace(['no', np.nan], '0', inplace=True)
    enc = preprocessing.LabelEncoder()
    df.degree = enc.fit_transform(df.degree)

    
    return df.drop('income', axis=1), df.income


# ### reading train and test files 

# In[3]:


df_train = pd.read_csv("tcd ml 2019-20 income prediction training (with labels).csv")
df_test = pd.read_csv("tcd ml 2019-20 income prediction test (without labels).csv")


# ### cleaning data

# In[4]:



df, income = cleaning(df_train.copy())
df_kag, income_kag = cleaning(df_test.copy())


# using target encoding on train set and fitting the same on test set
categorical = ['country', 'profession']
encoder = TargetEncoder(categorical).fit(df, income)

df = encoder.transform(df, income)
df_kag = encoder.transform(df_kag)


# ### transformation 

# In[5]:



sc_x = StandardScaler().fit(df)
df = sc_x.transform(df)
df_kag = sc_x.transform(df_kag)


# ### train-test split 

# In[6]:



X_train, X_test, y_train, y_test = model_selection.train_test_split(df, income, test_size=0.15)


# ### Train on catboostregressor 

# In[7]:



reg = CatBoostRegressor(iterations=2000, eval_metric='RMSE', depth=8, bagging_temperature=0.2, learning_rate=0.02)
reg.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model = True)


# ### predict and get RMSE

# In[8]:



# Test
y_pred = reg.predict(X_test)

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# ### publish results 

# In[9]:


predictions = reg.predict(df_kag)
pd.Series(predictions).to_csv('predictions.csv')

