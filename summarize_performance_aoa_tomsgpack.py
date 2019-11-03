#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib

df_aoa=pd.DataFrame()

with open('notfound.csv','w') as notfoundf:
    notfoundf.write('stage\tconv\n')
    for stage in list(range(0,10,2))+list(range(20,70,20)):
        for conv in range(1,6,1):
            lcpth='/home/ubuntu/linearclass_v3/linearclass_time_%d_conv_%d'%(stage,conv)
            print('Loading %s'%lcpth)
            aoapth=path.join(lcpth,'aoaresults_toplayer_epoch_1.json')  
            if os.path.exists(aoapth):
                print('Found %s'%aoapth)
                with open(aoapth,'r') as f:
                    aoa=json.load(f)
                    for key,val in aoa.items():
                        d={'epoch':[stage],'conv':[conv],'node':key,'aoa':val['aoa'],'loss':val['loss'],'prec5':val['prec5'],'prec1':val['prec1']}
                        df_aoa=df_aoa.append(pd.DataFrame.from_dict(d),ignore_index=True)
            else:
                notfoundf.write('%d\t%d\n'%(stage,conv))
                
df_aoa.to_json('linearclass_v3_aoa.json')
print(df_aoa)
