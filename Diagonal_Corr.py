# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 14:59:03 2018

@author: JSitters
"""

from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir(r'\\AA.AD.EPA.GOV\ORD\ATH\USERS\A-M\jsitters\Net MyDocuments\Research Documents\HMS Doc\Compare')

sns.set(style="white")

# Generate a dataset
hms=pd.read_csv('AZ_ncdcdaymetnldasgldas.csv')
labels=['NCDC', 'DAYMET', 'NLDAS', 'GLDAS']

rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = hms.corr(method='pearson') ##Need to add the number into the square

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots() #figsize=(11, 9)

plt.xlabel(labels)
plt.ylabel(labels)

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio  mask=mask, linewidths=.5, cbar_kws={"shrink": .5}   
sns.heatmap(corr,  cmap=cmap, mask=mask, vmax=1,  center=0.5,
            square=True , annot=True)


#%%
"""Cumulative Distribution Function"""

cumulative=np.cumsum(hms.iloc[:,1:])
#plt.legend()
plt.plot(cumulative)

#%%
"""Box Plots"""
plt.figure()
hms.boxplot()

#%%

"""Mean monthly Comparison Graph"""
hms.columns=['Date', 'NCDC', 'DAYMET', 'NLDAS', 'GLDAS']
pd.to_datetime(hms.Date)
hms.set_index('Date', inplace=True)

hms.plot(figsize=(5,5), linewidth=5, fontsize=20)
plt.xticks(rotation='vertical')
plt.xlabel('Month', fontsize=20);
