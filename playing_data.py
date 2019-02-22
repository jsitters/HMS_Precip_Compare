# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 10:37:55 2019

@author: JSitters
"""

import numpy as np
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from calendar import monthrange
import os
from sklearn.metrics import mean_absolute_error
sns.set(style="white")


#%%
path =r'O:\PRIV\WEB\HMS\Precip_Compare_Manuscript\Data\10yr_HMS_Ecoregion\data_without_metdata' 
#Create dataframe using pandas and import all csvs
df = pd.concat([pd.read_csv(f, sep=',',parse_dates=['Date'],
                            index_col=["Date"]) for f in glob.glob(path + "/*.csv")])
        #ncei
cols = ['ncdc', 'daymet', 'gldas', 'nldas', 'ecoregion'] 
df[cols] = df[cols].replace({'-1.00E+04':np.nan, -10000.0:np.nan})   #({ -9999.0:np.nan}) for full data
#df['2011-07-01'] #of pacific_nw is null

ecoregions=['Pacific NW', 'Pacific SW','Great Basin','Southwest', 'NRockies','SRockies','Mezquital', 'NPlains',  
'CPlains', 'SPlains','Prairie','Deep South','Souteast','Great Lakes', 'Appalachia', 'N Atlantic',  'MidAtlantic',]
  
"""Mean Monthly Precip"""
p=df.groupby([df.ecoregion])

missing=p['daymet'].count()-p['ncdc'].count()

what=df.groupby(['ecoregion',df.index.month]).sum()
ct=df.groupby(['ecoregion',df.index.month]).count() #days in month
std=df.groupby(['ecoregion',df.index.month]).std()
mon=what/11 #years

mon.plot( use_index=True, kind='bar')
plt.title("Mean Monthly Precipitation")
plt.ylabel(' Mean Monthly Precipitation (mm)')

  
"""Mean Monthly Precipitation"""    
fig = plt.figure(figsize=(20,10))

for eco,num in zip(mon.index.get_level_values(0).unique(), range(1,18)):
    df1 = mon.loc[eco]
    ax = fig.add_subplot(5,4,num)
    ax.plot( mon.loc[eco])#, kind='bar')#ax.bar(mon.loc[eco], height= )#error_kw=std.loc[eco]
    ax.set_title(eco, fontsize=20)
    ax.set_ylim(0,180) 
fig.subplots_adjust(hspace=0.1, wspace=0.1)    
plt.tight_layout()    
fig.title("Mean Monthly Precipitation")

"""Cumulative Sum graphs"""
yr=df.groupby(['ecoregion',df.index.year]).sum()
fig = plt.figure()
for eco,num in zip(yr.index.get_level_values(0).unique(), range(1,18)):
    df0 = yr.loc[eco]
    ax = fig.add_subplot(5,4,num)#, sharex=True)
    ax.plot(np.cumsum(df0))
    ax.set_title(eco)
    ax.set_xlabel('Precipitation')
    np.cumsum(yr)
fig.subplots_adjust(hspace=0.2, wspace=0.2) 
#%%
##Trying to use subplotS
fig, ax = plt.subplots(5,4, sharex=True, sharey=True)#mon.index.get_level_values(1).unique())

for eco,num, in zip(mon.index.get_level_values(0).unique(), range(1,18)):
    ax = fig.add_subplot(5,4,num)
    ax.plot( mon.loc[eco])#, kind='bar')#ax.bar(mon.loc[eco], height= )#error_kw=std.loc[eco]
    ax.set_title(eco)
    #ax.set_xlabel('Month') 
fig.subplots_adjust(hspace=0.4, wspace=0.4)    
#plt.tight_layout()    
plt.title("Mean Monthly Precipitation")

#for i in range(5):
#    for j in range(4):
#        for eco in zip(mon.index.get_level_values(0).unique()):#df0 = mon.loc[eco]   
total=p.apply(np.sum)

#os.chdir('//AA.AD.EPA.GOV/ORD/ATH/USERS/A-M/jsitters/Net MyDocuments/Research Documents/HMS Doc/Compare')
#
#
#hms=pd.read_csv('AZ_ncdcdaymetnldasgldas.csv', parse_dates=['Date'], index_col=["Date"],
#                na_values=[-10000.0, -9999.0], keep_default_na=False, verbose=True) #na_values just ignores the datapoint
#hms.columns= ['NCDC', 'DAYMET', 'NLDAS', 'GLDAS']
#
##Converting dtypes to numbers and dates !columns NLDAS and GLDAS were in numeric
#hms['DAYMET']=pd.to_numeric(hms['DAYMET'])
#hms['NCDC']=pd.to_numeric(hms['NCDC'])
#hms.index=pd.to_datetime(hms.index, format='%Y-%m-%d') #hms.index.str.split(" ")
####working with all the data for each ecoregion###
#labels=  ['NCEI','DAYMET','GLDAS','NLDAS','Ecoregion']
"""mean absolute error""" #no missing data
mae=pd.DataFrame(np.zeros(shape=(len(hms), 4) ))
mae.columns=hms.columns
mae.index+=1
i=0
for j in range(len(hms.columns)):
    if hms.notnull():
        mae.iloc[i][j]=mean_absolute_error(hms['NCDC'], hms[hms.columns[j]])
    i=i+1


