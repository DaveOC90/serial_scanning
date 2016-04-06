import glob
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

dct={}

for f in glob.glob('/data/ss_nifti/analysis_res/boxplots/*stats*'):
    temp=pd.read_csv(f)
    temp['mask']=['_'.join(f.split('_')[0:3]) for i in range(0,temp.shape[0])]
    dct[f]=temp
allstats=pd.concat(dct.values())
#allstats=allstats[allstats.type == 'diff']

allstats['state']=allstats.state1+allstats.state2

sns.set_context("poster")
plt.figure(figsize=(9,12))
sns.boxplot(x='type',y='mean',data=allstats[allstats.state=='allsubsallsubs'])

plt.savefig('/data/ss_nifti/analysis_res/boxplots/meancorr_groupmask.png',dpi=300)


dct={}

for f in glob.glob('/data/ss_nifti/analysis_res/varboxplots/*stats*'):
    temp=pd.read_csv(f)
    temp['mask']=['_'.join(f.split('_')[0:3]) for i in range(0,temp.shape[0])]
    dct[f]=temp
allstats=pd.concat(dct.values())
#allstats=allstats[allstats.type == 'diff']

allstats['state']=allstats.state1+allstats.state2

sns.set_context("poster")
plt.figure(figsize=(9,12))
sns.boxplot(x='type',y='mean',data=allstats[allstats.state=='allsubsallsubs'])

plt.savefig('/data/ss_nifti/analysis_res/varboxplots/meanvar_groupmask.png',dpi=300)
