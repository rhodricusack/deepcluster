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


def learningcurve(t,A,k):
    y=A*(1-(np.exp(-t*k)))
    return y

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
import pickle

clustering=pickle.load(open('clustering.pickle','rb'))

fig_hist,ax_hist=plt.subplots(ncols=5)
fig_scatter,ax_scatter=plt.subplots(ncols=5,sharex=True,sharey=True)

lc={}

df_kuperman.drop(columns='Unnamed: 2',inplace=True)

df_aoa['cluster']=df_aoa['node'].map(clustering['clusters'])

sel_epoch=60

cmap = plt.cm.get_cmap('tab20')


for convkey,convgrp in df_aoa.groupby('epoch').get_group(sel_epoch).groupby('conv'):
    print('Conv layer %d'%convkey)

    # Scatter plot of parameters againsts AoA, and histograms of parameters
    ax_hist[convkey-1].hist(convgrp['prec5'])
    sns.regplot(ax=ax_scatter[convkey-1], data=convgrp,x='aoa',y='prec5',marker=None,color='black')
    for clkey,clgrp in convgrp.groupby('cluster'):
        ax_scatter[convkey-1].scatter(data=clgrp,x='aoa',y='prec5',color=cmap(clkey/21.0))

    c=spearmanr(convgrp['aoa'],convgrp['prec5'])
    ax_scatter[convkey-1].set_xlabel('AoA (years)')
    ax_scatter[convkey-1].set_ylabel('Fully trained prec5')   
    print('Correlation aoa and epoch %d, spearman r=%f p<%f'%(sel_epoch,c[0],c[1]))



plt.show()