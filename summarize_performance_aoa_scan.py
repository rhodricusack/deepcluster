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

df_aoa=pd.DataFrame()
home = str(Path.home())

# Pull model from S3
s3 = boto3.resource('s3')

lcpth='rhodricusack/iclr2020/deepcluster_analysis/linearclass_v3'

with open('notfound.csv','w') as notfoundf:
    notfoundf.write('stage\tconv\n')
    for stage in list(range(0,10,2))+list(range(20,70,20)):
        for conv in range(1,6,1):
            # Create paths
            lcpth2='linearclass_time_%d_conv_%d'%(stage,conv)
            aoafn='aoaresults_toplayer_epoch_1.json'
            s3fn=path.join(lcpth,lcpth2,aoafn)  
            localpth=path.join(home,lcpth,lcpth2)
            localfn=path.join(localpth,aoafn)

            # Make directory to download file to
            if not path.exists(localpth):
                os.makedirs(localpth)

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

            # Process a file if we've received something
            if path.exists(localfn):
                print('Found %s'%localfn)
                with open(localfn,'r') as f:
                    aoa=json.load(f)
                    for key,val in aoa.items():
                        d={'epoch':[stage],'conv':[conv],'node':key,'aoa':val['aoa'],'loss':val['loss'],'prec5':val['prec5'],'prec1':val['prec1']}
                        df_aoa=df_aoa.append(pd.DataFrame.from_dict(d),ignore_index=True)

df_aoa.to_json('linearclass_v3_aoa.json')
print(df_aoa)
