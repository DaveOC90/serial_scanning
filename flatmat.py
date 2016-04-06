import os,glob,sys
import pandas as pd
import numpy as np

#sublist='/home/doconnor/ohbm16/sublist'
sublist='/home/doconnor/ohbm16/sublist_n3'
subs=[s.strip() for s in open(sublist,'rU')]

sldir='/home/doconnor/ohbm16/op_symlinks/ants_s0_g0/'



rois=sorted(['roi_aal_mask','roi_CC200_mask','roi_ho_mask','roi_random_k3200_mask','roi_rois_random_k0800','roi_rois_random_k0400','roi_rois_random_k1600'])

seshtoinclude=['ssc_2/','ssc_3/','ssc_4/','ssc_5/','ssc_6/','ssc_7/','ssc_9/','ssc_10/','ssc_11/','ssc_12/','ssc_13/','ssc_14/']

for roi in rois:
    flatcsv=pd.DataFrame()
    for root,dirs,fs in os.walk(sldir):
        for f in fs:
            if roi in f and any(x in root for x in seshtoinclude) and any(s in root for s in subs):
                 fpath=os.path.join(root,f)
                 print fpath
                 deets=fpath.replace(sldir,'').split('/')
                 subsesh=deets[0]
                 image=deets[1].split('_')[1]

                 colname='_'.join([subsesh,image,f.split('.')[0]])                    

                 data=pd.read_csv(fpath,index=0)
                 print data.shape
                 data=data.transpose()
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

    flatcsv.to_csv(roi+'n3_all_flatcorr.csv')
