import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import sys
import numpy as np
import os,glob
from natsort import natsorted
import natsort
import random

def corrdf_to_coldf(corrdf,keep_diag):

    mask = np.zeros_like(corrdf, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    if keep_diag == 'no':
        mask=np.invert(mask) ## Including this line will mean the diagonal of the corrmat will not be exported
    maskcorr=corrdf*mask
    maskcorr=maskcorr.unstack()

    return maskcorr

def plot_boxplots(coldf,opname):  
    fo=open(opname.replace('.png','_stats.txt'),'w')
    print  "###### Making Fields ######",opname
    coldf.columns = ['u1','u2','p']

    coldf['u1']=coldf.u1.str.split('/').str[-1]
    coldf['u2']=coldf.u2.str.split('/').str[-1]

    coldf['sub1']=coldf.u1.str.split('-').str[0]
    coldf['sub2']=coldf.u2.str.split('-').str[0]

    coldf['sesh1']=coldf.u1.str.split('-').str[1]
    coldf['sesh2']=coldf.u2.str.split('-').str[1]

    coldf['image1']=coldf.u1.str.split('-').str[2]
    coldf['image2']=coldf.u2.str.split('-').str[2]

    print  "###### Determining Within and Between Comparisons ######",opname
    coldf['type']=np.zeros(len(coldf))
    coldf.type[coldf.sub1 == coldf.sub2] = 'within'
    coldf.type[coldf.sub1 != coldf.sub2] = 'between'


    coldf=coldf[coldf.p != 0]

    print '### Loading QCMat'
    qcmat=pd.read_csv('/data/ss_nifti/analysis_res/cpac_motionop_qc.csv')

    ## Exclusion Criteria
    coldf=coldf[~coldf.u1.isin(list(qcmat['subseshscan'][qcmat.Signal == 0].values))]##Getting rid of scans with poor snr
    coldf=coldf[~coldf.u2.isin(list(qcmat['subseshscan'][qcmat.Signal == 0].values))]##Getting rid of scans with poor snr
    coldf=coldf[~coldf.u1.isin(list(qcmat['subseshscan'][qcmat.Registration == 0].values))]##Getting rid of scans with poor registration
    coldf=coldf[~coldf.u2.isin(list(qcmat['subseshscan'][qcmat.Registration == 0].values))]##Getting rid of scans with poor registration
    coldf=coldf[~coldf.u1.str.contains('ssc_8')]
    coldf=coldf[~coldf.u2.str.contains('ssc_8')]
    coldf=coldf[~coldf.u1.str.contains('ssc_1-')]
    coldf=coldf[~coldf.u2.str.contains('ssc_1-')]
    coldf=coldf[~coldf.u1.str.contains('M00475776')]
    coldf=coldf[~coldf.u2.str.contains('M00475776')]
    coldf=coldf[~coldf.u1.str.contains('M00448814')]
    coldf=coldf[~coldf.u2.str.contains('M00448814')]
    coldf=coldf[~coldf.u1.str.contains('M00421916')]
    coldf=coldf[~coldf.u2.str.contains('M00421916')]
    coldf=coldf[~coldf.u1.str.contains('M00499588-ssc_7-_scan_inscapes')]
    coldf=coldf[~coldf.u2.str.contains('M00499588-ssc_7-_scan_inscapes')]
    coldf=coldf[~coldf.u1.str.contains('M00499588-ssc_7-_scan_flanker')]
    coldf=coldf[~coldf.u2.str.contains('M00499588-ssc_7-_scan_flanker')]

    meanwithin=coldf.p[coldf.sub1 == coldf.sub2].mean()
    stdwithin=coldf.p[coldf.sub1 == coldf.sub2].std()
    meanbetween=coldf.p[coldf.sub1 != coldf.sub2].mean()
    stdbetween=coldf.p[coldf.sub1 != coldf.sub2].std()
    meandiff=meanwithin-meanbetween
    stddiff=stdwithin-stdbetween
    #print opname.split('/')[-1],'mean within',,'std within',
    #print opname.split('/')[-1],'mean between',,'std between',
    fo.write('state1,state2,type,mean,std\n')
    fo.write('allsubs,allsubs,within,'+str(meanwithin)+','+str(stdwithin)+'\n')
    fo.write('allsubs,allsubs,between,'+str(meanbetween)+','+str(stdbetween)+'\n')
    fo.write('allsubs,allsubs,diff,'+str(meandiff)+','+str(stddiff)+'\n')

    

    print "###### Plotting ######",opname
    f, axes = plt.subplots(1, 10, figsize=(20, 8))
    #sns.set_context("poster")
    thing=[['_scan_rest','_scan_rest'],['_scan_movie','_scan_movie'],['_scan_inscapes','_scan_inscapes'],['_scan_flanker','_scan_flanker'],['_scan_rest','_scan_movie'],['_scan_rest','_scan_inscapes'],['_scan_rest','_scan_flanker'],['_scan_movie','_scan_inscapes'],['_scan_movie','_scan_flanker'],['_scan_inscapes','_scan_flanker']]

    for j,i in enumerate(thing):
        sns.boxplot(x='type',y='p',data=coldf[((coldf.image1==i[0]) & (coldf.image2==i[1])) |  ((coldf.image2==i[0]) & (coldf.image1==i[1]))], order=['within','between'],color='red',ax=axes[j])
        sns.stripplot(x='type',y='p',data=coldf[((coldf.image1==i[0]) & (coldf.image2==i[1])) |  ((coldf.image2==i[0]) & (coldf.image1==i[1]))], order=['within','between'], jitter=True, size=2, linewidth=0,ax=axes[j])

        ## Stats
        withinmean=coldf.p[(((coldf.image1==i[0]) & (coldf.image2==i[1])) |  ((coldf.image2==i[0]) & (coldf.image1==i[1]))) & (coldf.type == 'within')].mean()
        withinstd=coldf.p[(((coldf.image1==i[0]) & (coldf.image2==i[1])) |  ((coldf.image2==i[0]) & (coldf.image1==i[1]))) & (coldf.type == 'within')].std()
        betweenmean=coldf.p[(((coldf.image1==i[0]) & (coldf.image2==i[1])) |  ((coldf.image2==i[0]) & (coldf.image1==i[1]))) & (coldf.type == 'between')].mean()
        betweenstd=coldf.p[(((coldf.image1==i[0]) & (coldf.image2==i[1])) |  ((coldf.image2==i[0]) & (coldf.image1==i[1]))) & (coldf.type == 'between')].std()
        meandiff=withinmean-betweenmean
        stddiff=withinstd-betweenstd
        fo.write(i[0]+','+i[1]+',within,'+str(withinmean)+','+str(withinstd)+'\n')
        fo.write(i[0]+','+i[1]+',between,'+str(betweenmean)+','+str(betweenstd)+'\n')
        fo.write(i[0]+','+i[1]+',diff,'+str(meandiff)+','+str(stddiff)+'\n')

        i=[ii.replace('_scan_','') for ii in i]
        axes[j].set_ylim([0,1])
        
        axes[j].set_title('-'.join(i), fontsize=14)
        axes[j]


    fo.close()
    print "###### Saving fig to "+opname+" ######"
    #plt.sup_title(opname.split('/')[-1].split('.')[0])
    plt.tight_layout()
    plt.savefig(opname,dpi=300)
    plt.close()
    plt.cla()

def foldcorrcorr(fold,flabel):
    bigdf=pd.DataFrame()
    for f in sorted(glob.glob(fold+'/*'+flabel+'*.csv')):
        name=f.split('/')[-1].replace('.csv','')
        print '-'.join( f.split('-')[:3])
        try:
            tempdf=pd.read_pickle(f)
        except KeyError:
            tempdf=pd.read_csv(f,index_col=0)
        bigdf['-'.join( f.split('-')[:3])]=tempdf[tempdf.columns[0]]
    bdfcorr=bigdf.corr()

    return bdfcorr

def rankcorr(bdfcorr):
    rankdf=pd.DataFrame()
    for col in sorted(natsort.natsorted(bdfcorr)):
        rankdf[col.split('/')[-1]]=bdfcorr.sort(col,ascending=False)[col].head(5).index.str.split('/').str[-1]
        print bdfcorr.sort(col,ascending=False)[col].head(5)

    rankdf.index=['match1','match2','match3','match4','match5']

    return rankdf

def rankcorr_rand(bdfcorr,scantype,subject):
    qcmat=pd.read_csv('/data/ss_nifti/analysis_res/cpac_motionop_qc.csv')

    excludelist=['ssc_8','ssc_1-','M00475776','M00448814','M00421916','M00499588-ssc_7-_scan_inscapes','M00499588-ssc_7-_scan_flanker']+list(qcmat['subseshscan'][qcmat.Signal == 0].values)+list(qcmat['subseshscan'][qcmat.Registration == 0].values)

    ## Exclusion Criteria
    bdfcorr=bdfcorr.drop([c for c in bdfcorr.columns if any (x in c for x in excludelist)],0)
    bdfcorr=bdfcorr.drop([c for c in bdfcorr.columns if any (x in c for x in excludelist)],1)

    bdfcorr=bdfcorr[[c for c in bdfcorr.columns if all(x in c for x in [scantype[0],subject])]]
    bdfcorr=bdfcorr.reindex([i for i in bdfcorr.index if scantype[1] in i])
    randimg=random.choice(bdfcorr.columns.values)    
    topconns=bdfcorr.sort(randimg,ascending=False)[randimg].head(2).index.str.split('/').str[-1]

    return topconns.values,randimg.split('/')[-1]

def rankmatchmat(bdfcorr):
    scans=[['_scan_rest','_scan_rest'],['_scan_movie','_scan_movie'],['_scan_inscapes','_scan_inscapes'],['_scan_flanker','_scan_flanker'],['_scan_rest','_scan_movie'],['_scan_rest','_scan_inscapes'],['_scan_rest','_scan_flanker'],['_scan_movie','_scan_inscapes']
,['_scan_movie','_scan_flanker'],['_scan_inscapes','_scan_flanker']]
    subs=['M00413068','M00416789','M00437261','M00440730','M00472509','M00477296','M00479682','M00483135','M00489505', 'M00499588']

    arr=np.zeros((len(scans),len(subs)))
    for i,sub in enumerate(subs):
        for j,scan in enumerate(scans):
            conns,scan=rankcorr_rand(bdfcorr,scan,sub)
            print conns[1],scan
            if conns[1].split('-')[0] == scan.split('-')[0]:
                arr[j,i]=1
            else:
                arr[j,i]=0

    print arr
    arr=arr.mean(1)
    print arr

    opdf=pd.DataFrame()
    for num,scan in enumerate(scans):
        opdf.set_value(scan[0].replace('_scan_',''),scan[1].replace('_scan_',''),arr[num])


    return opdf
            

def ranktoscore(rankdf):
    subdf=pd.DataFrame()
    sessiondf=pd.DataFrame()
    scandf=pd.DataFrame()

    for col in rankdf.columns:
        subdf[col]=rankdf[col].str.split('-').str[0] == col.split('-')[0]
        sessiondf[col]=rankdf[col].str.split('-').str[1] == col.split('-')[1]
        scandf[col]=rankdf[col].str.split('-').str[2] == col.split('-')[2]

    subdf.index=rankdf.index
    sessiondf.index=rankdf.index
    scandf.index=rankdf.index

    opdf=pd.DataFrame()
    opdf['subjectscore']=subdf.mean(1)
    opdf['sessionscore']=sessiondf.mean(1)
    opdf['scanscore']=scandf.mean(1)

    return opdf

analysisdir='/data/ss_nifti/analysis_res/'
stdcorrdir='/home/doconnor/ohbm16/std_spatial_corrmats/'

excludelist=[
'M00499588-ssc_7-_scan_flanker', \
'M00499588-ssc_7-_scan_inscapes' \
]

def run():
    for scorr in natsort.natsorted(os.listdir(stdcorrdir)):
       if 'CC200'in scorr: 
           scorr=os.path.join(stdcorrdir,scorr)
           roiname=scorr.split('/')[-1]
           print scorr, analysisdir

           ## Setup Filenames and Directories


           corrmatdir=os.path.join(analysisdir,'stdcorrmats')
           if not os.path.isdir(corrmatdir):
               os.makedirs(corrmatdir)
           corrmatfile=corrmatdir+'/'+roiname+'.csv'
           if not os.path.isfile(corrmatfile):
               print '###Calculating Correlation Matrix', analysisdir
               bdfcorr=foldcorrcorr(scorr,'')
               bdfcorr.to_csv(corrmatfile)
           else:
               print '###Correlation Matrix Already Exists, Loading'
               bdfcorr=pd.read_csv(corrmatfile,index_col=0)

           flatmatdir=os.path.join(analysisdir,'flatcorrmats')
           if not os.path.isdir(flatmatdir):
               os.makedirs(flatmatdir)
           maskcorrfile=os.path.join(flatmatdir,roiname+'.csv')
           if not os.path.isfile(maskcorrfile):
               print '###Masking and Flattening Corr Mat'
               maskcorr=corrdf_to_coldf(bdfcorr)
               maskcorr.to_csv(maskcorrfile)
               maskcorr=pd.read_csv(maskcorrfile,header=None)
           else:
               print '###Masked Correlation Matrix Already Exists, Loading'
               maskcorr=pd.read_csv(maskcorrfile,header=None)
               
           boxplotdir=os.path.join(analysisdir,'boxplots')
           if not os.path.isdir(boxplotdir):
               os.makedirs(boxplotdir)
           print '###Plotting', os.path.join(boxplotdir,roiname+'.png')
           plot_boxplots(maskcorr,os.path.join(boxplotdir,roiname+'.png'))


analysisdir='/data/ss_nifti/analysis_res/'
varcorrdir='/data/ss_nifti/rolling_spatial_corrmats/'



def run_variability():
    for scorr in natsort.natsorted(os.listdir(varcorrdir)):
       if 'CC200' in scorr: 
           scorr=os.path.join(varcorrdir,scorr)
           roiname=scorr.split('/')[-1]
           print scorr, analysisdir

           ## Setup Filenames and Directories


           corrmatdir=os.path.join(analysisdir,'varcorrmats')
           if not os.path.isdir(corrmatdir):
               os.makedirs(corrmatdir)
           corrmatfile=corrmatdir+'/'+roiname+'.csv'
           if not os.path.isfile(corrmatfile):
               print '###Calculating Correlation Matrix', analysisdir
               bdfcorr=foldcorrcorr(scorr, 'auc')
               bdfcorr.to_csv(corrmatfile)
           else:
               print '###Correlation Matrix Already Exists, Loading'
               bdfcorr=pd.read_csv(corrmatfile,index_col=0)

           flatmatdir=os.path.join(analysisdir,'flatvarcorrmats')
           if not os.path.isdir(flatmatdir):
               os.makedirs(flatmatdir)
           maskcorrfile=os.path.join(flatmatdir,roiname+'.csv')
           if not os.path.isfile(maskcorrfile):
               print '###Masking and Flattening Corr Mat'
               maskcorr=corrdf_to_coldf(bdfcorr,'no')
               maskcorr.to_csv(maskcorrfile)
               maskcorr=pd.read_csv(maskcorrfile,header=None)
           else:
               print '###Masked Correlation Matrix Already Exists, Loading: ',maskcorrfile
               maskcorr=pd.read_csv(maskcorrfile,header=None)
               
           boxplotdir=os.path.join(analysisdir,'varboxplots')
           if not os.path.isdir(boxplotdir):
               os.makedirs(boxplotdir)
           print '###Plotting', os.path.join(boxplotdir,roiname+'.png')
           plot_boxplots(maskcorr,os.path.join(boxplotdir,roiname+'.png'))





def create_predict_tables(ipdir):
    for corrdf in glob.glob(ipdir+'/*'):
        roiname=corrdf.split('/')[-1].split('.')[0]
        corrdata=pd.read_csv(corrdf,index_col=0)
        rankdf=rankcorr(corrdata)
        scores=ranktoscore(rankdf)
    
        strscores=scores.to_string()
        strscores=','.join([s for s in scores.to_string().split(' ') if s != ''])
        fo=open(analysisdir+'/'+roiname+'_predictscores_table.md','w')
        cols=strscores.split('\n')[0]
        fo.write('|'+cols.replace(',','|')+'|\n')
        underheader1="|:----"+''.join(["|:----" for col in cols.split(',')])+'|\n'
        underheader2='|'+''.join(["|" for col in cols.split(',')])+'|\n'
        fo.write(underheader1)
        fo.write(underheader2)
        for row in strscores.split('\n')[1:]:
            fo.write('|'+row.replace(',','|')+'|\n')
        fo.close()

def corrplots(ipfile,opname):
    opdir='/'.join(opname.split('/')[:-1])
    if not os.path.isdir(opdir):
        print "### Making Directory ####"
        os.makedirs(opdir)
    corrmat=pd.read_csv(ipfile,index_col=0)

    cols=list([c.split('/')[-1] for c in corrmat.columns])
    indices=list([i.split('/')[-1] for i in corrmat.index])

    corrmat.columns=[c.split('/')[-1] for c in corrmat.columns]
    corrmat.index=[i.split('/')[-1] for i in corrmat.index]

    cols=natsorted(cols,key= lambda s : s.split('-')[1])
    cols=natsorted(cols,key= lambda s : s.split('-')[2])
    cols=natsorted(cols,key= lambda s : s.split('-')[0])
    corrmat=corrmat[cols]

    indices=natsorted(indices,key= lambda s : s.split('-')[1])
    indices=natsorted(indices,key= lambda s : s.split('-')[2])
    indices=natsorted(indices,key= lambda s : s.split('-')[0])
    corrmat=corrmat.reindex(cols)

    print "###### Generating Heatmap ######"
    sns.heatmap(corrmat)
    print "###### Saving fig to "+opname+" ######"
    plt.savefig(opname)
    plt.title(opname.split('/')[-1].split('.')[0])
    plt.tight_layout()
    plt.close()
    plt.cla()

    return corrmat



def mpl_colors(lol):
    import matplotlib
    import matplotlib.cm as cm


    minima = min(min(lol))
    maxima = max(max(lol))

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)

    for i,l1 in enumerate(lol):
        for j,l2 in enumerate(l1):
            lol[i][j]=mapper.to_rgba(lol[i][j])

    return lol

def corrplots_from_flatmat(ipfile,opname):
     

    if not os.path.isfile(opname.replace('.png','.csv')):
        print '### Loading and unstacking corrmat'
        #flatmat=pd.read_csv('/data/ss_nifti/analysis_res/flatcorrmats/CC200_mask_2mm.csv',header=None)
        ## Above flatmat has no diagonal
        #corrmat=pd.read_csv('/data/ss_nifti/analysis_res/stdcorrmats/CC200_mask_2mm.csv',index_col=0)
        corrmat=pd.read_csv(ipfile,index_col=0)
        #flatmat=corrdf_to_coldf(corrmat, 'yes')
        flatmat=corrmat.unstack()

        flatmat.to_csv('temp.csv')
        flatmat=pd.read_csv('temp.csv',header=None)
        os.remove('temp.csv')
        

        print '### Loading QCMat'
        qcmat=pd.read_csv('/data/ss_nifti/analysis_res/cpac_motionop_qc.csv')

        flatmat.columns = ['u1','u2','p']

        flatmat['u1']=flatmat.u1.str.split('/').str[-1]
        flatmat['u2']=flatmat.u2.str.split('/').str[-1]

        flatmat=flatmat[~flatmat.u1.str.contains('unknown')]           
        flatmat=flatmat[~flatmat.u2.str.contains('unknown')]

        mdf=flatmat
        #mdf=pd.merge(flatmat,qcmat,left_on='u1',right_on='subseshscan',how='outer')
        print '##Excluding Stuff'
        ## Exclusion Criteria
        #mdf=mdf[mdf.p.notnull()] ##Getting rid of unmatched data
        mdf=mdf[~mdf.u1.isin(list(qcmat['subseshscan'][qcmat.Signal == 0].values))]##Getting rid of scans with poor snr
        mdf=mdf[~mdf.u2.isin(list(qcmat['subseshscan'][qcmat.Signal == 0].values))]##Getting rid of scans with poor snr
        mdf=mdf[~mdf.u1.isin(list(qcmat['subseshscan'][qcmat.Registration == 0].values))]##Getting rid of scans with poor registration
        mdf=mdf[~mdf.u2.isin(list(qcmat['subseshscan'][qcmat.Registration == 0].values))]##Getting rid of scans with poor registration
        mdf=mdf[~mdf.u1.str.contains('ssc_8')]
        mdf=mdf[~mdf.u2.str.contains('ssc_8')]
        mdf=mdf[~mdf.u1.str.contains('ssc_1-')]
        mdf=mdf[~mdf.u2.str.contains('ssc_1-')]
        mdf=mdf[~mdf.u1.str.contains('M00475776')]
        mdf=mdf[~mdf.u2.str.contains('M00475776')]
        mdf=mdf[~mdf.u1.str.contains('M00448814')]
        mdf=mdf[~mdf.u2.str.contains('M00448814')]
        mdf=mdf[~mdf.u1.str.contains('M00421916')]
        mdf=mdf[~mdf.u2.str.contains('M00421916')]
        mdf=mdf[~mdf.u1.str.contains('M00499588-ssc_7-_scan_inscapes')]
        mdf=mdf[~mdf.u2.str.contains('M00499588-ssc_7-_scan_inscapes')]
        mdf=mdf[~mdf.u1.str.contains('M00499588-ssc_7-_scan_flanker')]
        mdf=mdf[~mdf.u2.str.contains('M00499588-ssc_7-_scan_flanker')]

        mdf=mdf.reset_index()

        unqvals1=set(mdf.u1.values)
        unqvals2=set(mdf.u2.values)
  
        if len(unqvals1 - unqvals2) != 0:
            raise Exception('Columns dont match')

        print '###Sorting data'
        unqvalssort=natsorted(list(unqvals1),key=lambda s : s.split('-')[1])
        unqvalssort=natsorted(list(unqvalssort),key=lambda s : s.split('-')[0])
        unqvalssort=natsorted(list(unqvalssort),key=lambda s : s.split('-')[2])

        temp=zip(mdf.u1.values,[i for i in range(0,len(mdf.u1.values))],mdf.u2.values)

        temp=natsorted(temp,key=lambda s : s[2].split('-')[1])
        temp=natsorted(temp,key=lambda s : s[2].split('-')[0])
        temp=natsorted(temp,key=lambda s : s[2].split('-')[2])
        temp=natsorted(temp,key=lambda s : s[0].split('-')[1])
        temp=natsorted(temp,key=lambda s : s[0].split('-')[0])
        temp=natsorted(temp,key=lambda s : s[0].split('-')[2])

        img1,ind1,img2=zip(*temp)
        
        print '### Making new DF'
        newarr=np.zeros((np.sqrt(len(ind1)),np.sqrt(len(ind1))))
        ##Create New CorrMat
        for i,row in enumerate(ind1):
            #newdf.set_value(unqvalssort.index(mdf.u1.loc[row]),unqvalssort.index(mdf.u2[row]),mdf.p[row])
            #newdf.ix[mdf.u1.loc[row],unqvalssort.index(mdf.u2[row])]=mdf.p[row]
            newarr[unqvalssort.index(mdf.u1.loc[row]),unqvalssort.index(mdf.u2[row])]=mdf.p[row]
            print i,row

        newdf=pd.DataFrame(newarr,columns=unqvalssort,index=unqvalssort)
        newdf.to_csv(opname.replace('.png','.csv'))

    else:
        print 'Already have mat!!!'
        newdf=pd.read_csv(opname.replace('.png','.csv'),index_col=0)

    newdf=newdf.drop([c for c in newdf.columns if 'M00475776' in c],0)
    newdf=newdf.drop([c for c in newdf.columns if 'M00475776' in c],1)

    print '##Shape @@@@', newdf.shape

    newcols=[tuple(n) for n in newdf.columns.str.split('-')]
    newinds=[tuple(n) for n in newdf.index.str.split('-')]

    newdf.columns=pd.MultiIndex.from_tuples(newcols)
    newdf.index=pd.MultiIndex.from_tuples(newinds)

    subject_labels=newdf.columns.get_level_values(0)
    subject_pal = sns.light_palette('green',n_colors=subject_labels.unique().size)
    subject_lut = dict(zip(map(str, subject_labels.unique()), subject_pal))
    subject_colors = pd.Series(subject_labels).map(subject_lut)

    session_labels=newdf.columns.get_level_values(1)
    session_pal = sns.light_palette('blue',n_colors=session_labels.unique().size)
    session_lut = dict(zip(map(str, session_labels.unique()), session_pal))
    session_colors = pd.Series(session_labels).map(session_lut)

    scan_labels=newdf.columns.get_level_values(2)
    scan_pal = sns.light_palette('orange',n_colors=scan_labels.unique().size)
    scan_lut = dict(zip(map(str, scan_labels.unique()), scan_pal))
    scan_colors = pd.Series(scan_labels).map(scan_lut)

    rowcols=[subject_colors,session_colors,scan_colors]
    colcols=[subject_colors,session_colors,scan_colors]

    #plt.figure(figsize=(12,12))

    print "###### Generating Heatmap ######"
    #sns.set_context("poster")
    g=sns.clustermap(newdf, row_cluster=False, col_cluster=False,xticklabels=False,yticklabels=False,row_colors=rowcols,col_colors=colcols, vmin=0, vmax=1,cmap='Reds')

    #for label in subject_labels.unique():
    #    g.ax_col_dendrogram.bar(0, 0, color=subject_lut[label], label=label, linewidth=0)
    #    g.ax_col_dendrogram.legend(loc="center top", ncol=2)

    #for label in session_labels.unique():
    #    g.ax_col_dendrogram.bar(0, 0, color=session_lut[label], label=label, linewidth=0)
    #    g.ax_col_dendrogram.legend(loc="center upper", ncol=2)

    for label in scan_labels.unique():
        g.ax_col_dendrogram.bar(0, 0, color=scan_lut[label], label=label, linewidth=0)
        g.ax_col_dendrogram.legend(loc="upper center", ncol=2)


    opdir='/'.join(opname.split('/')[:-1])
    if not os.path.isdir(opdir):
        print "### Making Directory ####"
        os.makedirs(opdir)

    print "###### Saving fig to "+opname+" ######"
    #plt.title(opname.split('/')[-1].split('.')[0])
    #plt.tight_layout()
    #plt.gca().tight_layout()
    g.savefig(opname,dpi=300)
    plt.close()
    plt.cla()

def varclustplots():
    for i in glob.glob('/data/ss_nifti/analysis_res/varcorrmats/*CC200*.csv'):
        corrplots_from_flatmat(i,'/data/ss_nifti/analysis_res/varclusterplots/'+i.split('/')[-1].split('.')[0]+'.png')
def clustplots():
    for i in glob.glob('/data/ss_nifti/analysis_res/stdcorrmats/*CC200*.csv'):
        corrplots_from_flatmat(i,'/data/ss_nifti/analysis_res/clusterplots/'+i.split('/')[-1].split('.')[0]+'.png')
