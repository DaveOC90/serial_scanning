import os,glob,sys
import pandas as pd
import numpy as np

sublist='/home/doconnor/ohbm16/sublist'
subs=[s.strip() for s in open(sublist,'rU')]

sldir='/home/doconnor/ohbm16/concat_rest/'



rois=sorted(['roi_aal_mask','roi_CC200_mask','roi_ho_mask','roi_random_k3200_mask','roi_rois_random_k0800','roi_rois_random_k0400','roi_rois_random_k1600'])

for roi in rois:
    flatcsv=pd.DataFrame()
    for root,dirs,fs in os.walk(sldir):
        for f in fs:
            if roi in f and any(x in root for x in ['s2-7']) and any(s in root for s in subs):
                 fpath=os.path.join(root,f)
                 print fpath
                 deets=fpath.replace(sldir,'').split('/')
                 subsesh=deets[0]
                 image='restConcat'

                 colname='_'.join([subsesh,image,f.split('.')[0]])                    

                 data=pd.read_csv(fpath,index_col=0)
                 print data.shape
                 #data=data.transpose()
                 print data.shape
                 corr=data.corr()
                 print corr.shape
                 mask = np.zeros_like(corr, dtype=np.bool)
                 mask[np.triu_indices_from(mask)] = True
                 mask=np.invert(mask)
                 maskcorr=corr*mask
                 flat=np.ndarray.flatten(maskcorr.values)
                 flatnozeros=flat[flat!=0]
                 print len(flatnozeros)
                 flatcsv[colname]=flatnozeros

    flatcsv.to_csv(roi+'concatrest_flatcorr.csv')
