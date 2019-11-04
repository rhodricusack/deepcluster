
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load up
rdm=np.load('rdm.npy')
df_kuperman=pd.read_csv('kuperman_and_lastepoch.csv')
df_kuperman_sorted=df_kuperman.sort_values('layer4-epoch60')

# Show RDM
print('Show RDM')
plt.imshow(rdm)

synsetnames=list(df_kuperman_sorted['synset'])

# Clustering on RDM
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Dendrograms")  
Z=shc.linkage(rdm, method='ward')
t=0.1*np.max(Z[:,2])

dn=shc.dendrogram(Z,labels=synsetnames,color_threshold=t)
fc=shc.fcluster(Z,t=20,criterion='maxclust')

for clust in set(fc):
    print('Cluster %d'%clust)
    for item in np.where(fc==clust)[0]:
        print(synsetnames[item],end=' ')
    
    print('')

rdm2=rdm
for ind,leaf in enumerate(dn['leaves']):
    print(synsetnames[leaf])

# Reorder columns
for ind,leaf in enumerate(dn['leaves']):
    rdm2[ind,:]=rdm[leaf,:]

# Reorder rows
rdm3=rdm
for ind,leaf in enumerate(dn['leaves']):
    rdm3[:,ind]=rdm[:,leaf]

plt.figure()
plt.imshow(rdm3)


plt.show()





