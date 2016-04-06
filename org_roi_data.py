import glob,os,sys,subprocess
import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
from matplotlib import pyplot as plt
import natsort
tr=1.45

from scipy import sparse


def aucsims(A):
    return sp.integrate.simps(A,dx=1.45)
    

def return_labels_2d(ipdf):

    x,y=np.indices(ipdf.shape)
    x=np.char.array(ipdf.columns[x])
    y=np.char.array(ipdf.index[y])

    return x+'-'+y


def quick_df_load(dfpath):
    print 'Reading: ',dfpath,' as a textfile'
    thing=[l.strip() for l in open(dfpath,'rU')]
    print 'Creating array'
    vals=np.array([row.split(',')[1:] for row in thing[1:]])
    print 'Creating DataFrame'
    df=pd.DataFrame(vals,columns=thing[0].split(',')[1:])

    return df

def df_write_str(df,oppath):
    df=df.to_string()
    fo=open(oppath,'w')
    for row in df.split('\n'):
        fo.write(row+'\n')
    fo.close()

def pairwise_corr(A):

    #A=A.astype('float64')

    n=A.shape[0]
    
    xbar=A.mean(0)
    ybar=A.mean(0)

    xminusxbar=A-xbar
    yminusybar=A-ybar

    xminusxbarsquare=xminusxbar**2
    yminusybarsquare=yminusybar**2

    numer=np.dot(xminusxbar.T,yminusybar)
    denom=np.vstack(np.sqrt(xminusxbarsquare.sum(0)))*np.hstack(np.sqrt(yminusybarsquare.sum(0)))

    return numer/denom

def rolling_pairwise_corr(Aframe,opframe,window,serieslen):
    """
    A = Matrix with columns set to the series to be correlated
    window = size of window
    """

    Aframe=pd.read_csv(Aframe,index_col=0)
    Aframe=Aframe.T
    A=Aframe.values
    Anames=[str(col1)+'-'+str(col2) for col1 in Aframe.columns for col2 in Aframe.columns]
    Ashape=A.shape
    if not serieslen:
        serieslen=Ashape[0]
    else:
        pass #serieslen is as specified

    bigarr=np.zeros([Ashape[0]-(window-1),Ashape[1],Ashape[1]])

    print "Will produce matrix of size "+"x".join([str(serieslen-(window-1)),str(Ashape[1]),str(Ashape[1])])

    for i in range(0,serieslen-(window-1)):
        temparr=pairwise_corr(A[i:i+window,:])
        bigarr[i,:,:]=temparr
        print "running for window: "+str(i+1)+" out of "+str(serieslen-(window-1))

    bigshape=bigarr.shape
    bigarr=bigarr.reshape(bigshape[0],bigshape[1]*bigshape[2],order='C')
    
    bigdf=pd.DataFrame(bigarr,columns=Anames)

    return bigdf


def flat_corr_mat(ipdf):

    # Make columns labels roi num and row labels time series points
    # Then correlate rois
    corr=ipdf.transpose().corr()
    # make mask same shape as corrmat
    mask = np.zeros_like(corr, dtype=np.bool)
    #Take diagonal of mask, and upper triangle, and set to true
    mask[np.triu_indices_from(mask)] = True
    # Flip values in mask to include all values in lower triangle, minus the diagonal
    mask=np.invert(mask)
    # Flatten corrmat and include only values in mask
    flatcorr=np.ndarray.flatten(corr.values)
    flatmask=np.ndarray.flatten(mask)
    flatcorr=flatcorr[flatmask!= False]

    # Also create array of labels retained
    labels=return_labels_2d(corr)
    labels=np.ndarray.flatten(labels)
    labels=labels[flatmask!=False]

    return flatcorr, labels

def compare_lists(lst1,lst2):
     import collections
     return lambda x, y: collections.Counter(x) == collections.Counter(y)

def dict_to_csvs(dct,opdir,writestatus):
    if not os.path.isdir(opdir):
            os.makedirs(opdir)
    for k1 in dct.keys():
        opname=os.path.join(opdir,k1+'.csv')
        opdf=pd.DataFrame()
        for k2 in dct[k1].keys():
            opdf[k2]=dct[k1][k2]

        if writestatus == 'update' and os.path.isfile(opname):
            currdf=pd.read_csv(opname)
            if compare_lists(currdf.columns,opdf.columns) and all(np.array_equal(currdf[col],opdf[col]) for col in opdf.columns):
                print 'Dataframe exits and is identical'
            else:        
                opdf.to_csv(opname,index=False)
        else:
            opdf.to_csv(opname,index=False)


def rois_to_rollingcorr(ipcsv,opcsv,window):
    # Read Data and make each column a timeseries
    data=pd.read_csv(ipcsv,index_col=0)
    data=data.transpose()
    # Caclulate pairwise rolling correlation of columns (timeseries) using window specified
    t=pd.rolling_corr(data,window,pairwise=True)
    # Extract numpy array, create labels for rois corrleated (as col names) in C-like index order, and shape of array
    tvals=t.values
    tcols=[str(col)+'-'+str(row) for col in t['0'].columns for row in t['0'].index]
    tshape=tvals.shape
    # Reshape array so x indexes the correlations and y is the correlated timeseries, C-like order
    tvals=tvals.reshape(tshape[0],tshape[1]*tshape[2],order='C')
    # write to csv
    newdf=pd.DataFrame(tvals,columns=tcols)
    newdf.to_csv(opcsv)

    return newdf
        
def produce_corr_csvs(ipdir,opdir,labels,corrtype):
    #bigdict={}
    for ippath in glob.glob(os.path.join(ipdir,'*')):
        subsesh=ippath.split('/')[-1]
        
        sub=subsesh.split('_')[0]
        sesh='_'.join(subsesh.split('_')[1:])

        for root,dirs,fs in os.walk(ippath):
            for f in fs:
                if '.csv' in f and any(l in f for l in labels):#any(l in f for l in ['roi_rois_random_k0400_2mm.csv']):
                    fpath=os.path.join(root,f)
                    #print fpath
                    fpath_bits=fpath.split('/')
                    scanname=[fb for fb in fpath_bits if '_scan_' in fb]
                    if len(scanname) == 1:
                        scanname=scanname[0]
                    else:
                        scanname='unknown'

                    roiname=[fb for fb in fpath_bits if '_mask_' in fb]

                    if len(roiname) > 0:
                        roiname=roiname[0]
                    else:
                        roiname='unknown'


                    seshscanroi='-'.join([sesh,scanname,roiname])
                    subseshscanroi='-'.join([sub,sesh,scanname,roiname])

                    opcsv=os.path.join(os.path.abspath(opdir),'-'.join(labels),subseshscanroi+'.csv')
                    if not os.path.isfile(opcsv):

                        if corrtype == 'std':
                            tempdf=pd.DataFrame()
                            print 'Loading timeseries to dataframe: ',fpath
                            roidf=pd.read_csv(fpath,index_col=0)
                            print 'Generating and storing std correlation matrix: ',fpath
                            tempdf[seshscanroi],colnames=flat_corr_mat(roidf)
                            #bigdict.setdefault(sub,{})
                            #bigdict[sub].setdefault(seshscanroi,tempdf)
                            print 'Writing DataFrame: ',opcsv
                            if not os.path.isdir(os.path.join(os.path.abspath(opdir),'-'.join(labels))):
                                os.makedirs(os.path.join(os.path.abspath(opdir),'-'.join(labels)))
                            tempdf.to_csv(opcsv)
                        elif corrtype == 'rolling':
                            print 'Generating and writing pairwise rolling correlation matrix: ',fpath,'---->',opcsv
                            if not os.path.isdir(os.path.join(os.path.abspath(opdir),'-'.join(labels))):
                                os.makedirs(os.path.join(os.path.abspath(opdir),'-'.join(labels)))
                            rois_to_rollingcorr(fpath,opcsv,30)
                            #bigdict.setdefault(sub,{})
                            #bigdict[sub].setdefault(seshscanroi,tempdf)
                        elif corrtype == 'rolling_dave':
                            print 'Generating and storing rolling pairwise correlation matrix (davecode): ',fpath
                            tempdf=rolling_pairwise_corr(fpath,'',30,'')
                            #bigdict.setdefault(sub,{})
                            #bigdict[sub].setdefault(seshscanroi,tempdf)
                            print 'Writing DataFrame: ',opcsv
                            if not os.path.isdir(os.path.join(os.path.abspath(opdir),'-'.join(labels))):
                                os.makedirs(os.path.join(os.path.abspath(opdir),'-'.join(labels)))
                            tempdf.to_pickle(opcsv)
                            #df_write_str(tempdf,opcsv)

                    else:
                        print 'File already exists: ',opcsv

    #print 'Writing correlation matrices for: ',labels

    #opcsvdir=os.path.join(os.path.abspath(opdir),'-'.join(labels))
    #dict_to_csvs(bigdict,opcsvdir)

    #return bigdict

def generate_stdcorr_csvs():
    csvlist=['aal_mask_2mm','CC200_mask_2mm', 'ho_mask_2mm', 'shen_268_parcellation_2mm', 'rois_random_k0400_2mm', 'rois_random_k0800_2mm', 'rois_random_k1600_2mm', 'random_k3200_mask_2mm']
    #csvlist=['aal_mask_2mm','CC200_mask_2mm', 'ho_mask_2mm', 'rois_random_k0400_2mm']
    #csvlist=['random_k3200_mask_2mm']
    ipdir='/home/doconnor/ohbm16/op_symlinks/pipeline_serial_scanning/'
    opdir='/home/doconnor/ohbm16/std_spatial_corrmats'
    for cl in csvlist:
        produce_corr_csvs(ipdir,opdir,[cl],'std')

def generate_rollingcorr_csvs():
    #csvlist=['aal_mask_2mm','CC200_mask_2mm', 'ho_mask_2mm', 'rois_random_k0400_2mm', 'rois_random_k0800_2mm', 'shen_268_parcellation_2mm','random_k3200_mask_2mm', 'rois_random_k1600_2mm']
    #csvlist=['aal_mask_2mm','CC200_mask_2mm', 'ho_mask_2mm', 'rois_random_k0400_2mm']
    csvlist=['CC200_mask_2mm']
    ipdir='/home/doconnor/ohbm16/op_symlinks/pipeline_serial_scanning/'
    opdir='/data/ss_nifti/rolling_spatial_corrmats'
    for cl in csvlist:
        produce_corr_csvs(ipdir,opdir,[cl],'rolling_dave')

def parse_rollingcorrs():

    #Remove duplicate columns
    df=df[list(set(['-'.join(map(str,sorted(map(int,c.split('-'))))) for c in df.columns.values]))]
    scipy.integrate.simps(df[[0]].values.T,dx=1.45)

