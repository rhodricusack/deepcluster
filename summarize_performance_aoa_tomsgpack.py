#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib

df=pd.DataFrame(columns=['stage','conv','prec1','prec5','loss_log'])
df_aoa=pd.DataFrame(columns=['stage','conv','aoa','loss','node'])

for stage in [0,1]:
    for conv in range(0,4):
        lcpth='/home/rhodricusack/linearclass_v3/linearclass_time_%02d_conv_%d_v3'%(stage,conv)
        print('Loading %s'%lcpth)
        d={'stage':[stage],'conv':[conv]}
        df=df.append(pd.DataFrame.from_dict(d),ignore_index=True)
        aoapth=path.join(lcpth,'aoaresults.json')  
        with open(aoapth,'r') as f:
            aoa=json.load(f)
            for key,val in aoa.items():
                d={'stage':[stage],'conv':[conv],'node':key,'aoa':val['aoa'],'loss':val['loss']}
                df_aoa=df_aoa.append(pd.DataFrame.from_dict(d),ignore_index=True)

df_aoa.to_json('linearclass_v3_aoa.json')
print(df)
