#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib

df=pd.DataFrame(columns=['stage','conv','aoa','loss','node','prec1','prec5','loss_log'])

for stage in [0,1]:
    for conv in range(0,4):
        for toplayer_epoch in range(0,4):
            d={'stage':[stage],'conv':[conv],'toplayer_epoch':toplayer_epoch}
            suffix="_toplayer_epoch_%d"%toplayer_epoch
            lcpth='/home/rhodricusack/linearclass_v3/linearclass_time_%d_conv_%d_v3'%(stage,conv)
            print('Loading %s with suffix %s'%(lcpth,suffix))
            for item in ['prec1','prec5','loss_log']:
                with open(os.path.join(lcpth,item+suffix),'r') as f:
                    val=pickle.load(f)
                    d[item]=val
                    df=df.append(pd.DataFrame.from_dict(d),ignore_index=True)

df_aoa.to_json('linearclass_v3_toplayer_epoch.json')
print(df)
