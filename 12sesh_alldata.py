import pandas as pd
import os,glob


concatdir='/home/doconnor/ohbm16/flatcorr_restconcat/'
alldatadir='/home/doconnor/ohbm16/3sub_12sesh_flatcorr/'
opdatadir='/home/doconnor/ohbm16/flatcorr_rconc_withother/'

rois=sorted(['roi_aal_mask','roi_CC200_mask','roi_ho_mask','roi_random_k3200_mask','roi_rois_random_k0800','roi_rois_random_k0400','roi_rois_random_k1600'])


for alldata in glob.glob(alldatadir+'*flatcorr.csv'):
    for concrest in glob.glob(concatdir+'*12ses.csv'):
        for roi in rois:
        	if roi in alldata and roi in concrest and '3200' in roi:
        		print alldata.split('/')[-1], concrest.split('/')[-1]

        		alldf=pd.read_csv(alldata,index_col=0)
        		concrestdf=pd.read_csv(concrest,index_col=0)
        		alldf=alldf.drop([o for o in alldf.columns if 'Resting' in o],axis=1)
                        #alldf=alldf.drop([o for o in alldf.columns if not any(sub for sub in ['M00479682','M00489505','M00499588']] in o],axis=1)

        		opdf=pd.merge(alldf,concrestdf,left_index=True,right_index=True)

        		opdf.to_csv(opdatadir+roi+'_alldata_rest12ses.csv',index=False)
