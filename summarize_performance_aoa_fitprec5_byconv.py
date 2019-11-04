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
    return np.mean((model-prec5)**2) + k**2 + (A>100)*(A-100)**2   # now with regularisation to minimize k 

clustering=pickle.load(open('clustering.pickle','rb'))
cmap20 = plt.cm.get_cmap('tab20')


# Colormap for AoA
cmap = plt.cm.get_cmap('RdBu')
colors = cmap(np.arange(cmap.N))
print(cmap.N)


df_aoa=pd.read_json('/home/rhodricusack/deepcluster_analysis/linearclass_v3_aoa_toplayer_epoch_1.json')
df_kuperman=pd.read_csv('matchingAoA_ImageNet_excel.csv')

# Clusters to show semantics
df_aoa['cluster']=df_aoa['node'].map(clustering['clusters'])

for convkey,convgrp in df_aoa.groupby('cluster'):
    print("***cluster %d: "%convkey)
    for nodekey,nodegrp in convgrp.groupby('node'):
        krow=df_kuperman[df_kuperman['node']==nodekey]
        print('%s'%krow['synset'].iloc[0],end=' ')
    print("")
    print("")




df_aoa['aoa_rank']=df_aoa['aoa'].rank()

aoamin=df_aoa['aoa_rank'].max()
aoarange=df_aoa['aoa_rank'].min()-aoamin

from scipy.stats import spearmanr

# For timecourses
fig_tc,ax_tc=plt.subplots(ncols=2,figsize=(4,2),sharey=True)
fig_scatter,ax_scatter=plt.subplots(nrows=1,ncols=1,sharex=True,figsize=(3,3))


lc={}


# Just plot graph for conv 4, best performing deepcluster layer
sel_conv=4

for convkey,convgrp in df_aoa[df_aoa['conv']==sel_conv].groupby('conv'):
    print('Conv layer %d'%convkey)
    print(set(convgrp['epoch']))
    ax_tc[0].set_title('Conv layer %d'%convkey)
    aoa=[]
    lc={}
    for nodekey,nodegrp in convgrp.groupby('node'):

        # Fit decaying exponential curve to learning for each visual class 
        epoch=np.array([float(s) for s in nodegrp['epoch']])
        prec5=np.array([float(l) for l in nodegrp['prec5']])
        A0=float(nodegrp.iloc[-1]['prec5']) # starting estimate for A
        p=minimize(cost,[20,0])
        #, maxfev = 10000,bounds=((0,-5),(100,1))) 
        lc[nodekey]={'A':p.x[0],'k':p.x[1]}

        # Push back into dataframe
        df_aoa.loc[nodegrp.index,'A']=p.x[0]
        df_aoa.loc[nodegrp.index,'k']=p.x[1]
        

        # Plot learning curve for this class
        colind=int(((nodegrp['aoa_rank'].iloc[0]-aoamin)/aoarange)*255)
        ax_tc[0].plot(epoch,prec5,color=colors[colind],alpha=0.2)
        ax_tc[0].set_ylim([0,100])
        ax_tc[0].set_ylabel('prec5 (validation)')
        aoa.append(float(nodegrp['aoa'].iloc[0]))

        # Plot fits on separate chart
        prec5_fits=[learningcurve(s,p.x[0],p.x[1]) for s in epoch]
        colind=int(((nodegrp['aoa_rank'].iloc[0]-aoamin)/aoarange)*255)
        ax_tc[1].plot(epoch,prec5_fits,color=colors[colind],alpha=0.2)
        ax_tc[1].set_ylim([0,100])
        ax_tc[1].set_ylabel('fitted prec5 (validation)')



for convkey,convgrp in df_aoa[df_aoa['conv']==sel_conv].groupby('conv'):
    # Show scatter plots of parameters against AoA, with points coloured by semantic cateogry
    # Another loop so we pull in the new columns from the original dataframe
    for ind,parmname in enumerate(['A']):
        # Scatter plot of parameters againsts AoA, and histograms of parameters
        parm=[x[parmname] for k,x in lc.items()]
        #ax_hist[ind].hist(parm) 
        c=spearmanr(aoa,parm)

        ax_scatter.set_xlabel('AoA (years)')
        ax_scatter.set_ylabel('Fit parameter %s'%parmname)
        print('Correlation aoa and %s spearman r=%f p<%f'%(parmname,c[0],c[1]))

        sns.regplot(ax=ax_scatter, data=convgrp,x='aoa',y=parmname,marker=None,color='black')
        for clkey,clgrp in convgrp.groupby('cluster'):
            ax_scatter.scatter(data=clgrp,x='aoa',y=parmname,color=cmap20(clkey/21.0),label=clkey)
fig_scatter.savefig('deepcluster_AoAvsA.pdf')
ax_scatter.legend()
fig_scatter.savefig('deepcluster_AoAvsA_withlegend.pdf')

# See relationship between parameters for conv layers 0 and 3
conds=['A','k']
df=pd.DataFrame()
for node in lc:
    # Prepare for pairplot
    d={}
    for ind,cond in enumerate(conds):
        d['%s'%(cond)]=lc[node][cond]
    df=df.append(d,ignore_index=True)
sns_j=sns.jointplot(df['k'],y=df['A'],height=3)
c=spearmanr(df['k'],df['A'])
print('Correlation of k and A spearman r=%f p<%f'%(c[0],c[1]))

for node in lc:
    # Put fits into dataframe with AoA and node names, and write as csv
    for ind,cond in enumerate(conds):
        df_kuperman.loc[df_kuperman['node'] == node,'%s'%(cond)]=lc[node][cond]

df_kuperman.to_csv('kuperman_and_fits.csv')
sns_j.savefig('deepcluster_jointplot.pdf')
fig_tc.savefig('deepcluster_tc.pdf')
plt.show()
print(lc)






            

#%%
