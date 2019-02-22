# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 14:20:01 2019

@author: JSitters
"""
import numpy as np, scipy.stats as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
sns.set(style="white")
#import and read data files !needs to be updated to run through all filles in folder!
os.chdir('L:/Priv/WEB/HMS/Precip_Compare_Manuscript/Data/10yr_HMS_Ecoregion')
hms=pd.read_csv('13southeast_alabama.csv', parse_dates=['Date'], index_col=["Date"],
                na_values=[-10000.0, -9999.0], keep_default_na=False, verbose=True) #na_values just ignores the datapoint
hms.columns= ['NCDC', 'DAYMET', 'NLDAS', 'GLDAS']

#use only the data we need (not metadata)
end_date="2017-12-31 00"
hms=hms[(hms.index <=end_date)]

#Converting dtypes to numbers and dates !columns NLDAS and GLDAS were in numeric
hms['DAYMET']=pd.to_numeric(hms['DAYMET'])
hms['NCDC']=pd.to_numeric(hms['NCDC'])
hms.index=pd.to_datetime(hms.index, format='%Y-%m-%d') #hms.index.str.split(" ")


#loop through csv file
for t in range(len(hms)): #row
    for i in range(len(hms.columns)): #columns
    #missing data needs to be ignored
        if hms.iloc[t,i] < 0:
            hms.iloc[t,i]=0

stats=hms.describe()
tot=(hms.sum())
tot.name='Total Sum'
zero=(hms == 0).sum(axis=0)
zero.name='Zero Count'
stats.append([tot, zero])
    
#plots  
"""Mean Monthly Precip""" #add std error bar for each month
#agg= hms.resample('M', how='mean')#yr=agg.resample('A', how='mean')
group=hms.groupby([ hms.index.month]).mean()
std=hms.groupby([ hms.index.month]).std()

group.plot( kind='bar')
#group.plot.bar(yerr=std)#,stacked=True,)
#plt.xticks(['Jan', 'Feb', 'Mar', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(group.columns)
plt.xlabel('Month')
plt.ylabel(' Mean Monthly Precipitation (mm)')
 
"""Data Plot"""
fig, ax1=plt.subplots()
ax1.plot(hms.index.values, hms)
ax1.set_title('Precip at Location')

"""Box Plots"""
#had to add one to the data to take the log of it but zero data points are still outweighing and skewing the data
ax2 = plt.subplot()
#ax2.boxplot(np.log10(hms.values), labels=hms.columns)
ax2.boxplot(1+hms.values, labels=hms.columns)
ax2.set_yscale("log", basey=10)#, Log10Transform)#log scale
plt.ylim((10**-3, 100))
ax2.set_title('Precipitation BoxPlot')
ax2.set_ylabel('Log10 Precipitation')

"""Cumulative Sum"""
cumulative=np.cumsum(hms) 
plt.plot(cumulative)
plt.legend(('NCDC', 'DAYMET', 'NLDAS', 'GLDAS'))
plt.xlabel('Year')
plt.ylabel('Precipitation (mm)')
plt.title('Cumulative Sum')

"""Pearsons Corr"""
# Compute the correlation matrix
corr = hms.corr(method='pearson') ##Need to add the number into the square
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots() #figsize=(11, 9)
plt.xlabel(hms.columns)
plt.ylabel(hms.columns)
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio  mask=mask, linewidths=.5, cbar_kws={"shrink": .5}   
sns.heatmap(corr,  cmap=cmap, mask=mask, vmax=1,  center=0.5,
            square=True , annot=True)