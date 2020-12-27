# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 15:00:52 2020

@author: Swathi
"""

import Selenium_Scrappers as ss
import pandas as pd
path = "C:/Users/Swathi/Desktop/Swathi/ML/Project/Project_salary/chromedriver"
slp_time=15

df=ss.get_jobs('data scientist',100,False,path,slp_time)


