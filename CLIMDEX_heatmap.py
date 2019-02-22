# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 10:11:41 2019

@author: JSitters
"""

"""Count of Heavy Precip above a threshold for Ecoregions"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime
from sklearn.metrics import mean_absolute_error
import glob
sns.set(style="white")

path = os.chdir('L:/Priv/WEB/HMS/Precip_Compare_Manuscript/Data/36yr_HMS_Ecoregion')
files=sorted(glob.glob('*.csv'))

ecoregion=['Pacific NW', 'Pacific SW','Great Basin','Southwest', 'NRockies','SRockies','Mezquital', 'NPlains',  
'CPlains', 'SPlains','Prairie','Deep South','Souteast','Great Lakes', 'Appalachia', 'N Atlantic',  'MidAtlantic',]

end_date="2017-12-31 00"
cols=['ncei', 'daymet', 'nldas', 'gldas']

##create dataframes for specific calculations##
heat_map=pd.DataFrame(np.zeros(shape=(len(ecoregion), len(cols)) ))
mae=pd.DataFrame(np.zeros(shape=(len(ecoregion), len(cols)) ))
mx=pd.DataFrame(np.zeros(shape=(len(ecoregion), len(cols)) ))
wet=pd.DataFrame(np.zeros(shape=(len(ecoregion), len(cols)) ))
very=pd.DataFrame(np.zeros(shape=(len(ecoregion), len(cols)) ))
names=[heat_map, mae, mx, wet, very]
for name in names:   
    name.columns=cols
    name.index+=1


#### create dictionary of of each ecoregion csv dataFrames
regions=dict([(key, []) for key in ecoregion])
i=0
for f in files:
       
    df = pd.read_csv(f, index_col = 0) 
    df=df[(df.index <=end_date)]
    df=df[cols].apply(pd.to_numeric)
    df[cols] = df[cols].replace({ -9999.0:np.nan})
    ###For each ecoregion compute heatmap values

    
    for t in range(len(df)): #row
        for j in range(len(df.columns)): #columns
            mx.iloc[i][j]=df.max()[j]        
#            if df.iloc[t,j] >= 20:
#                very.iloc[i][j] += 1
#            elif df.iloc[t,j] >= 10: 
#               heat_map.iloc[i][j] += 1 
#            elif df.iloc[t,j] >= 1: #daymet only whole numbers
#                wet.iloc[i][j] += 1
    #creating the dictionary  
#    for j in range(len(df.columns)):
#            mae.iloc[i][j]=mean_absolute_error(df['ncdc'], df[df.columns[j]])      #need to show equations in  paper                
    #regions[ecoregion[i]].append(df) 
    i=i+1

"""detect_event""" 
#needs overall title, x,y titles, y ticklabels    
f, axes =plt.subplots(nrows=1, ncols=3)

cmap = sns.light_palette("purple", as_cmap=True)
sns.heatmap(wet,  cmap=cmap,  ax=axes[0],
            square=True , xticklabels=True, annot=False)  
axes[0].set_title('Wet Days >1mm')                      ##Need to find the >= sign
sns.heatmap(heat_map,  cmap=cmap, ax=axes[1], 
            square=True , xticklabels=True, annot=False) 
axes[1].set_title('Heavy Days >10mm')  
sns.heatmap(very,  cmap=cmap, ax=axes[2], 
            square=True , xticklabels=True, annot=False) #vmax=1000,
axes[2].set_title('Very Heavy Days >20mm')
#plt.title("CLIMDEX Indices")
#f.set_label('CLIMDEX Indices')                
labels=[ 'NCEI', 'DAYMET', 'NLDAS', 'GLDAS']


"""Max recorded"""
cmap = sns.light_palette("orange", as_cmap=True)#diverging_palette(220, 10, as_cmap=True)
sns.heatmap(mx,  cmap=cmap,  
            square=True , xticklabels=True, annot=False)
plt.title('Maximum Recorded Event')#Heavy Precipitation Days (>10mm)')
plt.xlabel('Precipitation Source')
#plt.yticks(ecoregion.value)
plt.ylabel('Ecoregion')


"""Mean Absolute Error"""
cmap = sns.light_palette("purple", as_cmap=True)#diverging_palette(220, 10, as_cmap=True)
sns.heatmap(mae.iloc[:,1:],  cmap=cmap,  
            square=True , xticklabels=True, annot=False)
plt.title('Mean Absolute Error')#Heavy Precipitation Days (>10mm)')
plt.xlabel('Precipitation Source')
#plt.yticks(ecoregion.value)
plt.ylabel('Ecoregion')
#%%
from sklearn.metrics import mean_absolute_error
path =r'O:\PRIV\WEB\HMS\Precip_Compare_Manuscript\Data\10yr_HMS_Ecoregion\data_without_metdata' 
#Create dataframe using pandas and import all csvs

df2 = pd.concat([pd.read_csv(f, sep=',') for f in glob.glob(path + "/*.csv")],ignore_index=True)
mae=mean_absolute_error(df['ncdc'], df['daymet'])
df2
df2.loc[df['ncdc'] == 'N/A','ncdc'] = np.nan                                                                                


plt.plot(df2['daymet'])
plt.show()
df2.plot(x='Date',y='ncdc')
plt.show()


