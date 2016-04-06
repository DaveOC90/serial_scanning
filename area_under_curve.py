import pandas as pd
import numpy as np
import os,glob
import scipy as sp
from org_roi_data import quick_df_load, aucsims
import sys

df_file=sys.argv[1]
measuretype=sys.argv[2]
remove_dups=sys.argv[3]

if measuretype not in ['mean','std','aucvals']:
    raise Exception('I cant do that!!!!!!')


if remove_dups not in ['yes','no']:
    raise Exception('Must be yes or no')

if remove_dups == 'no':
    dupevar='withdupes'
else:
    dupevar=''


if not os.path.isfile(df_file.split('.')[0]+'_'+measuretype+dupevar+'.csv'):
    #df_file='/data/ss_nifti/rolling_spatial_corrmats/rois_random_k0800_2mm/M00499588-ssc_3-_scan_rest-_mask_rois_random_k0800_2mm.csv'
    print df_file
    try:
        print 'Trying to read pickle'
        df=pd.read_pickle(df_file)
    except KeyError:
        print 'Didnt Work, reading as csv'
        df=quick_df_load(df_file)
        try:
            print 'Converting to float'
            df=df.astype('float')
        except ValueError:
            print 'Didnt work, removing first 29 rows and trying again'
            df=pd.DataFrame(df.values[29:,:].astype('float'),columns=df.columns)


    if remove_dups == 'yes':
        print 'Removing dublicate pairwise correlations'
        df=df[list(set(['-'.join(map(str,sorted(map(int,c.split('-'))))) for c in df.columns.values]))]


    if measuretype == 'aucvals':
        print 'Calculating area under curve'
        #sp.integrate.simps(df[[0]].values.T,dx=1.45)
        df=pd.DataFrame(df.values-df.values.mean(0),columns=df.columns,index=df.index)
        opdf=pd.DataFrame()
        opdf[df_file.split('/')[-1]]=np.apply_along_axis(aucsims,0,df)
    elif measuretype == 'mean':
        opdf=pd.DataFrame()
        opdf[df_file.split('/')[-1]]=np.apply_along_axis(np.mean,0,df)
    elif measuretype == 'std':
        opdf=pd.DataFrame()
        opdf[df_file.split('/')[-1]]=np.apply_along_axis(np.std,0,df)

    print 'Writing to pickle: ',df_file.split('.')[0]+'_'+measuretype+dupevar+'.csv'
    opdf.to_pickle(df_file.split('.')[0]+'_'+measuretype+dupevar+'.csv')

else:
    print 'File Already Exists!!'
