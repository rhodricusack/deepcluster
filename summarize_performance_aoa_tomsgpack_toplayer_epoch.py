#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib

df=pd.DataFrame(columns=['stage','conv','toplayer_epoch','prec1','prec5','loss_log'])

for stage in range(0,70,20):
    for conv in range(1,6,1):
        for toplayer_epoch in range(5):
            d={'stage':[stage],'conv':[conv],'toplayer_epoch':[toplayer_epoch]}
            suffix="_toplayer_epoch_%d"%toplayer_epoch
            lcpth='/home/ubuntu/linearclass_v3/linearclass_time_%d_conv_%d'%(stage,conv)
            for item in ['prec1','prec5','loss_log']:
                pth=os.path.join(lcpth,'log',item+suffix)
                if os.path.exists(pth):
                    print('Loading %s'%(pth))
                    with open(pth,'rb') as f:
                        val=pickle.load(f)
                        d[item]=float(val[-1])
                else:
                    print('Not found %s'%pth)
            if 'prec1' in d.keys():
                df=df.append(pd.DataFrame.from_dict(d),ignore_index=True)

df.to_json('linearclass_v3_toplayer_epoch.json')
print(df)


#%%
fig,ax=plt.subplots(ncols=4,sharey=True,sharex=True)
for epochkey,epochgrp in df.groupby('stage'):
    for trainkey,traingrp in epochgrp.groupby('conv'):
        print(traingrp)
        print(epochkey)
        if epochkey % 20==0:
            ax[int(epochkey/20)].plot(1+traingrp['toplayer_epoch'],traingrp['prec5'],label="%d"%trainkey)
    ax[int(epochkey/20)].set_xlim([1,5])
    ax[int(epochkey/20)].set_xlabel('Epochs of top layer training')
    ax[int(epochkey/20)].set_ylabel('Top-5 precision')
    ax[int(epochkey/20)].set_title('Conv epochs %d'%epochkey)
plt.legend()