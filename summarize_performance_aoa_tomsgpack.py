#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib

df_aoa=pd.DataFrame()


for stage in range(70):
    for conv in range(1,6,1):
        lcpth='/home/ubuntu/linearclass_v3/linearclass_time_%d_conv_%d'%(stage,conv)
        print('Loading %s'%lcpth)
        aoapth=path.join(lcpth,'aoaresults.json')  
        if os.path.exists(aoapth):
            print('Found %s'%aoapth)
            with open(aoapth,'r') as f:
                aoa=json.load(f)
                for key,val in aoa.items():
                    d={'epoch':[stage],'conv':[conv],'node':key,'aoa':val['aoa'],'loss':val['loss'],'prec5':val['prec5'],'prec1':val['prec1']}
                    df_aoa=df_aoa.append(pd.DataFrame.from_dict(d),ignore_index=True)

df_aoa.to_json('linearclass_v3_aoa.json')
print(df_aoa)
