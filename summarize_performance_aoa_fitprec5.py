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
from scipy.optimize import minimize

def learningcurve(t,A,k):
    y=A*(1-(np.exp(-t*k)))
    return y

def cost(params):
    A, k =params
    model = learningcurve(epoch, A, k)
    return np.mean((model-prec5)**2) + k**2 + (A-50)**2    # minimize k 

# Colormap for AoA
cmap = plt.cm.get_cmap('RdBu')
colors = cmap(np.arange(cmap.N))
print(cmap.N)


df_aoa=pd.read_json('/home/rhodricusack/deepcluster_analysis/linearclass_v3_aoa_toplayer_epoch_1.json')

df_kuperman=pd.read_csv('matchingAoA_ImageNet_excel.csv')

# print('Dropping epoch 0')
#df_aoa=df_aoa[df_aoa.epoch != 0]
#df_aoa=df_aoa[df_aoa.epoch != 5]
df_aoa['aoa_rank']=df_aoa['aoa'].rank()

aoamin=df_aoa['aoa_rank'].max()
aoarange=df_aoa['aoa_rank'].min()-aoamin

from scipy.stats import spearmanr

fig,ax=plt.subplots(ncols=5,sharey=True)
figfits,axfits=plt.subplots(ncols=5,sharey=True)

fig_hist,ax_hist=plt.subplots(nrows=3,ncols=5)
fig_scatter,ax_scatter=plt.subplots(nrows=3,ncols=5,sharex=True)
fig_c0vs3,ax_c0vs3=plt.subplots(ncols=1)

lc={}

# for nodekey,nodegrp in df_aoa.groupby('node'):
#     colind=int(((nodegrp['aoa_rank'].iloc[0]-aoamin)/aoarange)*255)
#     convgroups=nodegrp.groupby('conv')
#     c0=convgroups.get_group(1)
#     c5=convgroups.get_group(5)
#     ax_c0vs3.plot(c0['prec5'],c5['prec5'],color=colors[colind],alpha=0.4)
#     ax_c0vs3.set_xlabel('conv 0 prec5')
#     ax_c0vs3.set_ylabel('conv 5 prec5')
#     ax_c0vs3.axis('equal')


for convkey,convgrp in df_aoa.groupby('conv'):
    print('Conv layer %d'%convkey)
    ax[convkey-1].set_title('Conv layer %d'%convkey)
    aoa=[]
    lc[convkey-1]={}
    for nodekey,nodegrp in convgrp.groupby('node'):

        # Fit decaying exponential curve to learning for each visual class 
        epoch=np.array([float(s) for s in nodegrp['epoch']])
        prec5=np.array([float(l) for l in nodegrp['prec5']])
        A0=float(nodegrp.loc[nodegrp['epoch']==60]['prec5']) # starting estimate for A
        p=minimize(cost,[20,0])
        #, maxfev = 10000,bounds=((0,-5),(100,1))) 
        lc[convkey-1][nodekey]={'A':p.x[0],'k':p.x[1]}


        # Plot learning curve for this class
        colind=int(((nodegrp['aoa_rank'].iloc[0]-aoamin)/aoarange)*255)
        ax[convkey-1].plot(epoch,prec5,color=colors[colind],alpha=0.2)
        ax[convkey-1].set_ylim([0,100])
        ax[convkey-1].set_ylabel('prec5 (validation)')
        aoa.append(float(nodegrp['aoa'].iloc[0]))

        # Plot fits on separate chart
        prec5_fits=[learningcurve(s,p.x[0],p.x[1]) for s in epoch]
        colind=int(((nodegrp['aoa_rank'].iloc[0]-aoamin)/aoarange)*255)
        axfits[convkey-1].plot(epoch,prec5_fits,color=colors[colind],alpha=0.2)
        axfits[convkey-1].set_ylim([0,100])
        axfits[convkey-1].set_ylabel('fitted prec5 (validation)')


    for ind,parmname in enumerate(['A','k']):
        # Scatter plot of parameters againsts AoA, and histograms of parameters
        parm=[x[parmname] for k,x in lc[convkey-1].items()]
        ax_hist[ind][convkey-1].hist(parm) 
        ax_scatter[ind][convkey-1].scatter(x=aoa,y=parm,s=2 )
        c=spearmanr(aoa,parm)
        ax_scatter[ind][convkey-1].set_xlabel('AoA (years)')
        ax_scatter[ind][convkey-1].set_ylabel('Fit parameter %s'%parmname)
        print('Correlation aoa and %s spearman r=%f p<%f'%(parmname,c[0],c[1]))

# See relationship between parameters for conv layers 0 and 3
conds=[[0,'A'],[0,'k'],[4,'A'],[4,'k']]
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
