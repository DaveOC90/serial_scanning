import glob,os,sys,subprocess
import pandas as pd
import numpy as np
import scipy as sp
from scipy import sparse
import seaborn as sns
from org_roi_data import pairwise_corr

serieslen=100
window=30

testdata=pd.read_csv('/home/doconnor/ohbm16/op_symlinks/pipeline_serial_scanning/M00499588_ssc_6/roi_timeseries/_scan_inscapes/_mask_aal_mask_2mm/roi_aal_mask_2mm.csv',index_col=0)

testdata=testdata.T
testdata=testdata.iloc[0:serieslen]


#testdata=np.array([np.arange(0,serieslen,1) if i % 2 == 0 else np.array(list(reversed(np.arange(0,serieslen,1)))) for i in range(0,100)])
#testdata=pd.DataFrame(testdata.T,columns=['col'+str(i) for i in np.arange(0,100,1)])


tshape=testdata.shape
bigarr=np.zeros([tshape[0]-(window-1),tshape[1],tshape[1]])
print bigarr.shape
#bigarrpandas=np.zeros([tshape[1]-30,tshape[0],tshape[0]])

for i in range(0,serieslen-(window-1)):
    temparr=pairwise_corr(testdata.values[i:i+window,:])
    bigarr[i,:,:]=temparr
#    bigarrpandas=testdata.T.iloc[i:i+30].corr().values
    print 'mycorr: ',i




pdarr=pd.rolling_corr(testdata,window,pairwise=True)
pdarr=pdarr[(window-1):,:,:]


print bigarr.shape,pdarr.shape
t=np.abs(bigarr)-np.abs(pdarr.values)
print t.max(), t.min(), t.mean(), t.std()
#print bigarr-bigarrpandas
