import pandas as pd
import numpy as np
import os,glob
from matplotlib import pyplot as plt
import seaborn as sns
from org_roi_data import quick_df_load, aucsims


rollmatdir='/data/ss_nifti/rolling_spatial_corrmats/'
analysisdir='/data/ss_nifti/analysis_res/'
qcmat=pd.read_csv('/data/ss_nifti/analysis_res/cpac_motionop_qc.csv')
for i in sorted(os.listdir(rollmatdir)):
    i=os.path.join(rollmatdir,i)
    print i
    samfile=glob.glob(i+'/*_2mm.csv')[0]
    print samfile
    try:
        colnames=pd.read_pickle(samfile)
    except:
        colnames=quick_df_load(samfile)
        
    colnames=colnames.columns.values
    for val in ['meanwithdupes','stdwithdupes']:#,'aucvalswithdupes']:
        for img in ['_scan_rest','_scan_flanker','_scan_movie','_scan_inscapes']:

            if not os.path.isfile((os.path.join(analysisdir,'gaogroupmeanplots/')+opname.replace('.png','.csv')):
                tdict={}
                for f in glob.glob(i+'/*'+img+'*'+val+'.csv'):
                    tdict[f.split('/')[-1].split('.')[0]]=pd.read_pickle(f).T
                tdf=pd.concat(tdict.values())
                tdf['scans']=tdict.keys()

                print 'Excluding'
                tdf=tdf[~tdf.scans.str.contains('ssc_8')]
                tdf=tdf[~tdf.scans.str.contains('ssc_1-')]
                tdf=tdf[~tdf.scans.str.contains('M00475776')]
                tdf=tdf[~tdf.scans.str.contains('M00448814')]
                tdf=tdf[~tdf.scans.str.contains('M00421916')]
                tdf=tdf[~tdf.scans.str.contains('M00499588-ssc_7-_scan_inscapes')]
                tdf=tdf[~tdf.scans.str.contains('M00499588-ssc_7-_scan_flanker')]
                tdf=tdf[~tdf.scans.str.contains('|'.join(list(qcmat['subseshscan'][qcmat.Registration == 0].values)))]
                tdf=tdf[~tdf.scans.str.contains('|'.join(list(qcmat['subseshscan'][qcmat.Signal == 0].values)))]
                tdf=tdf.drop('scans',1)

                tdf.columns=colnames


                labels1=sorted(list(set([t.split('-')[0] for t in colnames])))
                labels2=list(reversed(sorted(list(set([t.split('-')[1] for t in colnames])))))

                hmap=np.zeros((len(labels1),len(labels2)))
   
                print 'Making Heatmap'
                for k,l1 in enumerate(labels1):
                    for j,l2 in enumerate(labels2):
                        hmap[k,j]=tdf[l1+'-'+l2].mean()

                opname='-'.join([i.split('/')[-1],val,img])+'.png'

                newdf=pd.DataFrame(hmap,columns=labels1,index=labels2)
                print 'Writing Corrmat, ',os.path.join(analysisdir,'gaogroupmeanplots/')+opname.replace('.png','.csv')
                newdf.to_csv(os.path.join(analysisdir,'gaogroupmeanplots/')+opname.replace('.png','.csv'))

            else:
                newdf=pd.read_csv(os.path.join(analysisdir,'gaogroupmeanplots/')+opname.replace('.png','.csv'),index_col=0)

            #opname='-'.join([i.split('/')[-1],val,img])+'.png'
            opdir=os.path.join(analysisdir,'gaogroupmeanplots/')
            if not os.path.isdir(opdir):
                os.makedirs(opdir)

            print 'Plotting and saving'
            sns.heatmap(newdf)
            plt.savefig(os.path.join(opdir,opname))
#            plt.show()
            plt.close()
            plt.cla()
