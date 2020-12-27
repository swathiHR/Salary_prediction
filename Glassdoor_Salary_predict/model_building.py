# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 23:40:34 2020

@author: Swathi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('EDA_cleaned.csv')

#Choose required columns

df_model=df[['avaerage_salary','Rating','Size','Type of ownership','Industry', 'Sector','Revenue','hourly', 'employer_provided','job_location','age', 'python_yn','spark_yn', 'aws_yn', 'excel_yn', 'job_simp', 'Seniority',
       'desc_len']]
#get dummy data
df_model=pd.get_dummies(df_model)
#train test split
x=df_model.drop('avaerage_salary',axis=1)
y=df_model.avaerage_salary.values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#mutiple linear regression
import statsmodels.api as sm
X_sm=sm.add_constant(x)
model=sm.OLS(y,X_sm)
model.fit().summary()


from sklearn.linear_model import LinearRegression,Lasso
lm=LinearRegression()
lm.fit(X_train,y_train)

from sklearn.model_selection import cross_val_score
cross_val_score(lm,X_train,y_train,scoring = 'neg_mean_absolute_error',cv=3)
#lasso regression
lm_lasso=Lasso(alpha=234)
np.mean(cross_val_score(lm_lasso,X_train,y_train,scoring = 'neg_mean_absolute_error',cv=3))

alpha=[]
error=[]

for i in range(100,1000):
    alpha.append(i)
    lml=Lasso(alpha=(i))
    error.append(np.mean(cross_val_score(lml,X_train,y_train,scoring = 'neg_mean_absolute_error',cv=3)))
    

plt.plot(alpha,error)

err=tuple(zip(alpha,error))
df_err=pd.DataFrame(err,columns=['alpha','error'])
df_err[df_err.error==max(df_err.error)]
    
#random Forest
from sklearn.ensemble import RandomForestRegressor
rfr=RandomForestRegressor()
np.mean(cross_val_score(rfr,X_train,y_train,scoring = 'neg_mean_absolute_error',cv=3))

#tune models using gridsearchcv

from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'),'max_features':('auto','sqrt','log2')}
gs=GridSearchCV(rfr,parameters,scoring = 'neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_



#test ensembles
ypred_lm=lm.predict(X_test)
ypred_lasso=lm_lasso.predict(X_test)
y_pred_rf=gs.best_estimator_.predict(X_test)



from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test,ypred_lm)
mean_absolute_error(y_test,ypred_lasso)
mean_absolute_error(y_test,y_pred_rf)