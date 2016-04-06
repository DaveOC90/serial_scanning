import os, glob, sys, shutil
import pandas as pd

sldir='/home/doconnor/ohbm16/op_symlinks/ants_s0_g0/'

subdict={}
flist=[]

rois=sorted(['roi_aal_mask.csv','roi_CC200_mask.csv','roi_ho_mask.csv','roi_random_k3200_mask.csv','roi_rois_random_k0800.csv','roi_rois_random_k0400.csv','roi_rois_random_k1600.csv'])
flist.append(['subid','seshid','image','fname']+rois)

for subsesh in os.listdir(sldir):
    for image in os.listdir(os.path.join(sldir,subsesh)):
        for deriv in os.listdir(os.path.join(sldir,subsesh,image)):
            if 'roi' in deriv:
                sub,sesh=subsesh.split('-')
                print sub,sesh,image,deriv
                csvs=[c for c in os.listdir(os.path.join(sldir,subsesh,image,deriv)) if '.csv' in c]
                preslist=[]
                for r in rois:
                    if r in csvs:
                        preslist.append(1)
                    else:
                        preslist.append(0)

                flist.append([sub,sesh,image,deriv]+preslist)       

opflist=pd.DataFrame(data=flist[1:],columns=flist[0])
opflist.to_csv('test.csv')

