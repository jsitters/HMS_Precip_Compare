y# -*- coding: utf-8 -*-
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
path =r'L:\Priv\WEB\HMS\Precip_Compare_Manuscript\Data\36yr_HMS_Ecoregion\data_without_metadata' 
df = pd.concat([pd.read_csv(f, sep=',', parse_dates=['Date'],
                            index_col=["Date"]) for f in glob.glob(path + "/*.csv")])

cols = ['ncei', 'daymet', 'gldas', 'nldas','PRISM', 'ecoregion'] 
df[cols] = df[cols].replace({-10000.0:np.nan, -9999.0:np.nan})#{'-1.00E+04':np.nan, -10000.0:np.nan})   # for full data
#df['2011-07-01'] #of pacific_nw is null

ecoregion=['Pacific NW', 'Pacific SW','Great Basin','Southwest', 'NRockies','SRockies','Mezquital', 'NPlains',  
'CPlains', 'SPlains','Prairie','Deep South','Southeast','Great Lakes', 'Appalachia', 'N Atlantic',  'MidAtlantic',]
  
#Grouping the dataframe by ecoregion
p=df.groupby([df.ecoregion])
df.to_csv('all_data.csv')
missing=p['daymet'].count()-p['ncei'].count() #to get % divide by number of days and *100

sums=df.groupby(['ecoregion',df.index.month]).sum()
#ct=df.groupby(['ecoregion',df.index.month]).count() 
std=df.groupby(['ecoregion',df.index.month]).std()
mon=sums/37 #years 


  
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
fig.title("Mean Monthly Precipitation (mm)")

"""Cumulative Sum graphs"""
yr=df.groupby(['ecoregion',df.index.year]).sum()
fig = plt.figure(figsize=(20,10))
for eco,num in zip(yr.index.get_level_values(0).unique(), range(1,18)):
    df0 = yr.loc[eco]
    ax = fig.add_subplot(5,4,num)#, sharex=True)
    ax.plot(np.cumsum(df0))
    ax.set_title(eco)
    #ax.set_xlabel('Precipitation')
    np.cumsum(yr)
fig.subplots_adjust(hspace=0.2, wspace=0.2)

"""Maximum Value Recorded"""
cmap = sns.light_palette("red", as_cmap=True)#diverging_palette(220, 10, as_cmap=True)
sns.heatmap(p.max(),  cmap=cmap,  
            square=True , xticklabels=True, annot=False)

"""Pearsons Corr"""
# Compute the correlation matrix
corr = p.corr(method='pearson') ##Need to add the number into the square
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots() #figsize=(11, 9)
for eco,num in zip(yr.index.get_level_values(0).unique(), range(1,18)):
    sns.heatmap(corr.loc[eco],  cmap=cmap, mask=mask,  vmax=1,  center=0.5,
            square=True , annot=True)
    plt.xlabel(cols[:-1])
    plt.ylabel(cols[:-1])
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio  mask=mask, linewidths=.5, cbar_kws={"shrink": .5}   
sns.heatmap(corr,  cmap=cmap, mask=mask, vmax=1,  center=0.5,
            square=True , annot=True)


"""Mean Absolute Error"""
i=0
mae=pd.DataFrame(np.zeros(shape=(len(ecoregion), len(cols)) ))
#mae.columns=cols
mae.index+=1
for eco in (df['ecoregion'].unique()):
    if df['ecoregion']==eco:
        for j in range(len(df.columns)-1): 
            dna=df.dropna(axis=0) #Does not work for NAN values must skip
            mae.iloc[i][j]=dna.groupby(['ecoregion']).apply([mean_absolute_error(dna['ncei'], dna[dna.columns[j]]) for j in range(len(df.columns)-1)]) #dna.groupby(['ecoregion']).apply(\
            mae[i][j+1].append(eco)
    i=i+1
cmap = sns.light_palette("red", as_cmap=True)#diverging_palette(220, 10, as_cmap=True)
sns.heatmap(mae.iloc[:,1:],  cmap=cmap,  
            square=True , xticklabels=True, annot=False)
plt.title('Mean Absolute Error')
plt.xlabel('Precipitation Source')
#plt.yticks(ecoregion.value)
plt.ylabel('Ecoregion')        


df.groupby(["ecoregion"])[["ncei", "daymet","gldas","nldas"]].apply(mean_absolute_error()) #needs x and y argument
#%%
##Trying to use subplotS
fig, ax = plt.subplots(5,4, sharex=True, sharey=True)#mon.index.get_level_values(1).unique())

for eco,num, in zip(mon.index.get_level_values(0).unique(), range(1,18)):
    ax = fig.add_subplot(5,4,num)
    ax=(sns.heatmap(corr,  cmap=cmap, mask=mask, vmax=1,  center=0.5,
            square=True , annot=True))
    ax.set_title(eco)
    #ax.set_xlabel('Month') 
fig.subplots_adjust(hspace=0.4, wspace=0.4)    
#plt.tight_layout()    
plt.title("Mean Monthly Precipitation")

#for i in range(5):
#    for j in range(4):
#        for eco in zip(mon.index.get_level_values(0).unique()):#df0 = mon.loc[eco]   
total=p.apply(np.sum)





