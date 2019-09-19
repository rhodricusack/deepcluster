#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib

df=pd.read_json('../deepcluster_analysis/deepcluster_performance_summary.json')
df=df.sort_values(by='epoch')

#%%
plt.figure()
fig,ax=plt.subplots()
for key, grp in df.groupby(['conv']):
    grp.plot(ax=ax,kind='line', x='epoch', y='prec5',  label=key)
ax.set_xlabel('epoch')
ax.set_ylabel('prec5 (validation)')

#%%

df_prec5_wide=df.pivot(index='epoch',columns='conv',values='prec5')
plt.figure()
plt.imshow(df_prec5_wide)

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np


fig = plt.figure()
ax = fig.gca(projection='3d')

verts=[]
zs=[]
for epoch,epochgrp in df.groupby('epoch'):
    zs.append(epoch)
    verts.append(list(zip(epochgrp['conv'],epochgrp['prec5'])))


def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

facecolors=[cc('r'), cc('g'), cc('b'),cc('y')]*int(1+len(verts)/4)
poly = PolyCollection(verts, facecolors=facecolors[:len(verts)])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('Layer')
ax.set_xlim3d(0, 6)
ax.set_ylabel('Epoch')
ax.set_ylim3d(0, 80)
ax.set_zlabel('prec5')
ax.set_zlim3d(0, 50)

#%%
plt.figure()
fig,ax=plt.subplots()
for key, grp in df.groupby(['conv']):
    grp.plot(ax=ax,kind='line', x='epoch', y='loss_log',  label=key)
ax.set_ylim([0,7])
ax.set_xlabel('epoch')
ax.set_ylabel('loss (validation)')

plt.show()


plt.show()


#%%
from scipy.stats import pearsonr

fig,ax=plt.subplots()
for key, grp in df_aoa.groupby(['epoch']):
    lossbylayer={}
    plt.figure()
    for nodekey, nodegrp in grp.groupby(['node']):
        nodegrp.plot(ax=ax,x='conv', y='loss')
ax.set_xlabel('conv')
ax.set_ylim([0,10])
ax.set_ylabel('loss (validation)')
plt.show()


            

#%%
