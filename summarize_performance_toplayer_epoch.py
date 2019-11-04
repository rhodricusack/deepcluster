#%% 

import os
from os import path
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import json
import matplotlib

import boto3
import botocore

from pathlib import Path

df=pd.DataFrame(columns=['stage','conv','toplayer_epoch','prec1','prec5','loss_log'])

lcpth='rhodricusack/iclr2020/deepcluster_analysis/linearclass_v3'

with open('notfound_toplayer_epoch.csv','w') as notfoundf:
    for stage in list(range(0,10,2))+list(range(20,70,20)):
        for conv in range(1,6,1):
            for toplayer_epoch in range(5):
                d={'stage':[stage],'conv':[conv],'toplayer_epoch':[toplayer_epoch]}
                suffix="_toplayer_epoch_%d"%toplayer_epoch
                lcpth2='linearclass_time_%d_conv_%d'%(stage,conv)
                localpth=path.join(home,lcpth,lcpth2)
                # Make directory to download file to
                if not path.exists(localpth):
                    os.makedirs(localpth)

                for item in ['prec1','prec5','loss_log']:
                    itemfn=item+suffix
                    s3fn=path.join(lcpth,lcpth2,itemfn)  
                    localfn=path.join(localpth,itemfn)
                    
                    # Download a file if not already present locally
                    if not path.exists(localfn):
                        try:
                            print("Attempting s3 download from %s"%s3fn )
                            s3.Bucket('neurana-imaging').download_file(s3fn,localfn)
                        except botocore.exceptions.ClientError as e:
                            if e.response['Error']['Code'] == "404" or e.response['Error']['Code'] == "403" :
                                print("The object does not exist.")
                                notfoundf.write('%d\t%d\n'%(stage,conv))
                            else:
                                raise
                    # Load it up
                    if os.path.exists(localfn):
                        print('Loading %s'%(localfn))
                        with open(localfn,'rb') as f:
                            val=pickle.load(f)
                            d[item]=float(val[-1])
                if 'prec1' in d.keys():
                    df=df.append(pd.DataFrame.from_dict(d),ignore_index=True)

df.to_json('linearclass_v3_toplayer_epoch.json')
print(df)


