# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 16:44:56 2020

@author: Swathi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('data_glassdoor.csv')

#salary parsing
dataset=dataset[dataset['Salary Estimate'] != '-1']
salary=dataset['Salary Estimate'].apply(lambda x : x.split('(')[0])
minus_kd=salary.apply(lambda x : x.replace('k',' ').replace('$',''))

dataset['hourly']=dataset['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
dataset['employer_provided']=dataset['Salary Estimate'].apply(lambda x: 1 if 'employee provided salary' in x.lower() else 0)
min_hr=minus_kd.apply(lambda x : x.lower().replace('per hour',' ').replace('employee provided salary', ' '))
dataset['min_salary']=min_hr.apply(lambda x: int(x.split('-')[0]))
dataset['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
dataset['avaerage_salary']=(dataset['min_salary']+dataset['max_salary'])/2


#company name

dataset['company_txt']=dataset['Company Name'].apply(lambda x : x[0:-3])
dataset['job_location']=dataset['Location'].apply(lambda x : x.split(',')[0])
dataset['age']=dataset['Founded'].apply(lambda x: x if x<0 else 2020-x)

#job dscription

dataset['python_yn'] = dataset['Job Description'].apply(lambda x : 1 if 'python' in x.lower() else 0)
dataset['rstudio'] = dataset['Job Description'].apply(lambda x : 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
dataset['spark_yn'] = dataset['Job Description'].apply(lambda x : 1 if 'spark' in x.lower() else 0)
dataset['aws_yn'] = dataset['Job Description'].apply(lambda x : 1 if 'aws' in x.lower() else 0)
dataset['excel_yn'] = dataset['Job Description'].apply(lambda x : 1 if 'excel' in x.lower() else 0)

data_out=dataset.drop(['Index'],axis=1)
data_out.to_csv('Salary_cleaned.csv',index=False)

df=pd.read_csv('Salary_cleaned.csv')