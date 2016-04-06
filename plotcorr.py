import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import sys
import numpy as np
import os

ip=sys.argv[1]


if not os.path.isfile(ip.split('.')[0]+'maskcorr_unstacked.csv'):
    print ip, "Corr, Mask and Unstack"
    data=pd.read_csv(ip)
    corr=data.corr()
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    mask=np.invert(mask)
    maskcorr=corr*mask

    maskcorr=maskcorr.unstack()

    maskcorr.to_csv(ip.split('.')[0]+'maskcorr_unstacked.csv')



#test=pd.read_csv(f.split('.')[0]+'maskcorr_unstacked.csv',header=None)
test=pd.read_csv(ip.split('.')[0]+'maskcorr_unstacked.csv',header=None)
print ip, "###### Making Fields ######"
test.columns = ['u1','u2','p']

test['sub1']=test.u1.str.split('_').str[0]
test['sub2']=test.u2.str.split('_').str[0]

test['sub1']=test.sub1.str.split('-').str[0]
test['sub2']=test.sub2.str.split('-').str[0]

test['sesh1']=test.u1.str.split('_').str[1]
test['sesh2']=test.u2.str.split('_').str[1]

test['image1']=test.u1.str.split('_').str[2]
test['image2']=test.u2.str.split('_').str[2]

print ip, "###### Determining Within and Between Comparisons ######"
test['type']=np.zeros(len(test))
test.type[test.sub1 == test.sub2] = 'within'
test.type[test.sub1 != test.sub2] = 'between'

test=test[test.p != 0]

test.to_csv('joke.csv')
#sns.boxplot(x='type',y='p',data=test,color='red')
#sns.stripplot(x='type',y='p',data=test, jitter=True, size=2, linewidth=0)
#plt.show()



print ip, "###### Plotting ######"
f, axes = plt.subplots(1, 10, figsize=(20, 8))

thing=[['rest','rest'],['Movie','Movie'],['Inscapes','Inscapes'],['Flanker','Flanker'],['rest','Movie'],['rest','Inscapes'],['rest','Flanker'],['Movie','Inscapes'],['Movie','Flanker'],['Inscapes','Flanker']]
for j,i in enumerate(thing):
    sns.boxplot(x='type',y='p',data=test[((test.image1==i[0]) & (test.image2==i[1])) |  ((test.image2==i[0]) & (test.image1==i[1]))], order=['within','between'],color='red',ax=axes[j])
    sns.stripplot(x='type',y='p',data=test[((test.image1==i[0]) & (test.image2==i[1])) |  ((test.image2==i[0]) & (test.image1==i[1]))], order=['within','between'], jitter=True, size=2, linewidth=0,ax=axes[j])
    axes[j].set_ylim([0,1])
    axes[j].set_title('-'.join(i))

plt.savefig('plots/'+ip.split('.')[0]+'boxplot.png')
plt.title(ip.split('.')[0])
plt.tight_layout()
plt.close()
plt.cla()



