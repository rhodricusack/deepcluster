#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from  nltk.corpus import wordnet as wn
from nltk.metrics import edit_distance

import nltk
nltk.download('wordnet_ic')

from nltk.corpus import wordnet_ic

import pickle

brown_ic = wordnet_ic.ic('ic-brown.dat')

df_kuperman=pd.read_csv('matchingAoA_ImageNet_excel.csv')

# Get list of synsets
print('Getting list of synsets')
ss=[]

for ind,row in df_kuperman.iterrows():
    wn.synset
    ss.append(wn.synset_from_pos_and_offset('n', int(row['node'][1:])))

# Calculate RDM for semantics 
print('Calculating RDM')
nss=len(ss)
rdm=np.full((nss,nss),np.nan)
for ind1,ss1 in enumerate(ss):
    for ind2,ss2 in enumerate(ss):
        if ind2<=ind1:
            rdm[ind1,ind2]=wn.lch_similarity(ss1,ss2,brown_ic)
            rdm[ind2,ind1]=rdm[ind1,ind2]

#%%



# Clustering on RDM
import scipy.cluster.hierarchy as shc

Z=shc.linkage(rdm, method='ward')

maxclust=20

fc=shc.fcluster(Z,t=maxclust,criterion='maxclust')

clustering={'clusters':{},'cluster_method':'fcluster','criterion':'maxclust','number of clusters':maxclust}
for ind,row in df_kuperman.iterrows():
    clustering['clusters'][row['node']]=fc[ind]
    
pickle.dump(clustering,open('clustering.pickle','wb'))




    

