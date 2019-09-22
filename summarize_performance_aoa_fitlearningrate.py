#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import seaborn as sns
import math
import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from scipy.stats import spearmanr


def learningcurve(t,A,k,B):
    y=B+A*(np.exp(-t*k))
    return y

# Colormap for AoA
cmap = plt.cm.get_cmap('RdBu')
colors = cmap(np.arange(cmap.N))
print(cmap.N)


df_aoa=pd.read_json('/home/rhodricusack/deepcluster_analysis/linearclass_v2_aoa.json')

df_kuperman=pd.read_csv('matchingAoA_ImageNet_excel.csv')

print('Dropping epoch 0')
#df_aoa=df_aoa[df_aoa.stage != 0]
#df_aoa=df_aoa[df_aoa.stage != 5]
df_aoa['aoa_rank']=df_aoa['aoa'].rank()

aoamin=df_aoa['aoa_rank'].max()
aoarange=df_aoa['aoa_rank'].min()-aoamin

from scipy.stats import spearmanr

fig,ax=plt.subplots(ncols=4,sharey=True)
figfits,axfits=plt.subplots(ncols=4,sharey=True)

fig_hist,ax_hist=plt.subplots(nrows=3,ncols=4)
fig_scatter,ax_scatter=plt.subplots(nrows=3,ncols=4,sharex=True)
fig_c0vs3,ax_c0vs3=plt.subplots(ncols=1)

lc={}

for nodekey,nodegrp in df_aoa.groupby('node'):
    colind=int(((nodegrp['aoa_rank'].iloc[0]-aoamin)/aoarange)*255)
    convgroups=nodegrp.groupby('conv')
    c0=convgroups.get_group(1)
    c5=convgroups.get_group(5)
    ax_c0vs3.plot(c0['loss'],c5['loss'],color=colors[colind],alpha=0.4)
    ax_c0vs3.set_xlabel('conv 0 loss')
    ax_c0vs3.set_ylabel('conv 5 loss')
    ax_c0vs3.axis('equal')


for convkey,convgrp in df_aoa.groupby('conv'):
    print('Conv layer %d'%convkey)
    ax[convkey].set_title('Conv layer %d'%convkey)
    aoa=[]
    lc[convkey]={}
    for nodekey,nodegrp in convgrp.groupby('node'):

        # Fit decaying exponential curve to learning for each visual class 
        stage=np.array([float(s) for s in nodegrp['stage']])
        loss=np.array([float(l) for l in nodegrp['loss']])
        A0=float(nodegrp.loc[nodegrp['stage']==0]['loss']) # starting estimate for A
        B0=float(nodegrp.loc[nodegrp['stage']==50]['loss']) # starting estimate for B
        p=curve_fit(learningcurve,stage,loss,p0=[A0-B0,0,B0], maxfev = 10000) 
        lc[convkey][nodekey]={'A':p[0][0],'k':p[0][1],'B':p[0][2]}


        # Plot learning curve for this class
        colind=int(((nodegrp['aoa_rank'].iloc[0]-aoamin)/aoarange)*255)
        ax[convkey].plot(stage,loss,color=colors[colind],alpha=0.4)
        ax[convkey].set_ylim([0,10])
        ax[convkey].set_ylabel('loss (validation)')
        aoa.append(float(nodegrp['aoa'].iloc[0]))

        # Plot fits on separate chart
        loss_fits=[learningcurve(s,p[0][0],p[0][1],p[0][2]) for s in stage]
        colind=int(((nodegrp['aoa_rank'].iloc[0]-aoamin)/aoarange)*255)
        axfits[convkey].plot(stage,loss_fits,color=colors[colind],alpha=0.4)
        axfits[convkey].set_ylim([0,10])
        axfits[convkey].set_ylabel('fitted loss (validation)')


    for ind,parmname in enumerate(['A','B','k']):
        # Scatter plot of parameters againsts AoA, and histograms of parameters
        parm=[x[parmname] for k,x in lc[convkey].items()]
        ax_hist[ind][convkey].hist(parm) 
        ax_scatter[ind][convkey].scatter(x=aoa,y=parm,s=2 )
        c=pearsonr(aoa,parm)
        ax_scatter[ind][convkey].set_xlabel('AoA (years)')
        ax_scatter[ind][convkey].set_ylabel('Fit parameter %s'%parmname)
        print('Correlation aoa and %s r=%f p<%f'%(parmname,c[0],c[1]))

# See relationship between parameters for conv layers 0 and 3
conds=[[0,'B'],[0,'k'],[3,'B'],[3,'k']]
df=pd.DataFrame()
for node in lc[0]:
    # Prepare for pairplot
    d={}
    for ind,cond in enumerate(conds):
        d['%d %s'%(cond[0],cond[1])]=lc[cond[0]][node][cond[1]]
    df=df.append(d,ignore_index=True)
plt.figure()
sns.pairplot(df,kind='reg',diag_kind='kde',markers='.')

for node in lc[0]:
    # Put fits into dataframe with AoA and node names, and write as csv
    for ind,cond in enumerate(conds):
        df_kuperman.loc[df_kuperman['node'] == node,'%d %s'%(cond[0],cond[1])]=lc[cond[0]][node][cond[1]]

df_kuperman.to_csv('kuperman_and_fits.csv')
plt.show()
print(lc)






            

#%%
