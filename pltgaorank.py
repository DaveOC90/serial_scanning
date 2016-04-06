import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
import scipy as sp
import glob

#masks=['aal_mask_2mm','CC200_mask_2mm', 'ho_mask_2mm', 'shen_268_parcellation_2mm', 'rois_random_k0400_2mm', 'rois_random_k0800_2mm']
masks=['aal_mask_2mm']

for mask in masks:
    for calc in ['mean']:#['mean','std']:
        csvs=glob.glob('*'+mask+'*'+calc+'*.csv')
        bdct={}
        for c in csvs:
            bdct[c]=pd.read_csv(c,index_col=0)
        alldata=pd.concat(bdct.values())
        all_rank=sp.stats.rankdata(alldata.values,'average')/(alldata.shape[0]*alldata.shape[1])

        for keynum,key in bdct.keys():
            currdf=bdct[key]
            ind1=0+currdf.shape[1]
            ind2=ind1+currdf.shape[1]
            newdf=pd.DataFrame(all_rank[ind1,ind2],columns=currdf.columns,index=currdf.indices)
            bdct[key]=newdf
            print mask,calc,key
            sns.heatmap(newdf)
            plt.show()


