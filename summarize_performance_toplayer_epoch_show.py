import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib

df=pd.read_json('../deepcluster_analysis/linearclass_v3_toplayer_epoch.json')

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

fig.savefig('deepcluster_toplayer_epochs.pdf')

plt.show()
