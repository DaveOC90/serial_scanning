import os,glob,sys
import pandas as pd
import numpy as np

sublist='/home/doconnor/ohbm16/sublist'
subs=[s.strip() for s in open(sublist,'rU')]

sldir='/home/doconnor/ohbm16/op_symlinks/ants_s0_g0/'
concatdir='/home/doconnor/ohbm16/concat_rest/'

images=['Rest']

flatcsv=pd.DataFrame()

for sub in subs:
    for image in images:
        csvs=glob.glob(sldir+sub+'-ssc_[2-7,9]/*'+image+'*/roi_ts/*')+glob.glob(sldir+sub+'-ssc_[1][0-9]/*'+image+'*/roi_ts/*')
        csvnames=set([c.split('/')[-1] for c in csvs])
        for cn in csvnames:
            conc=sorted([c for c in csvs if cn in c])
            if len(conc) == 12:
                clist=[]
                for i,c in enumerate(conc):
                    temp=pd.read_csv(c)
                    temp=temp.drop(temp.columns[0],1).transpose()
                    clist.append(temp)

                opc=pd.concat(clist)
                if not os.path.isdir(concatdir+sub+'/s2-7_8-14/'+image):
                    os.makedirs(concatdir+sub+'/s2-7_8-14/'+image)
 
                opc.to_csv(concatdir+sub+'/s2-7_8-14/'+image+'/rest_s2-7_8-14_concat'+cn.split('.')[0]+'.csv')
            else:
                print sub, image, cn, "Dont have sessions 2-7 8-14....skip"


