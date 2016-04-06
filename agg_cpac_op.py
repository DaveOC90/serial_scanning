import glob
import os
import sys
import subprocess
import pandas as pd


def token_set(iplist,token):
    opdict={}
    for c1,supstr in enumerate(iplist):
        sublist=[i for i in iplist if i != supstr]
        difflist=[]
        for c2,substr in enumerate(sublist):
            #print c1,'/',len(iplist),c2,'/',len(sublist)
            tempdiffs=list(set(supstr.split(token))-set(substr.split(token)))
            difflist=difflist+tempdiffs
        opdict[supstr]=list(set(difflist))

    return opdict



global_direc='/data/cpac_warehouse/'
preproc_direc=global_direc
sublist_file='/home/doconnor/ohbm16/sublist'
regdir='output/'
sldir='/home/doconnor/ohbm16/op_symlinks/'
name_key=pd.read_csv('/home/doconnor/ohbm16/output_key_abridged.csv')



derivs=list(name_key.cpac_op)
#derivs=['falff_to_standard_smooth','reho_to_standard_smooth','vmhc_raw_score', 'centrality_outputs_zstd_smoothed','dr_tempreg_maps_files_to_standard_smooth','functional_mni']

sublist=sorted([s.strip() for s in open(sublist_file, 'rU')])

pipes=glob.glob(preproc_direc+regdir+'pipeline*/')
for pipe in pipes:
    pipename=pipe.replace(preproc_direc+regdir, '')
    for subdir in glob.glob(os.path.join(pipe,'*')):
        subdirname=subdir.replace(pipe,'')
        for deriv in derivs:
            temp=[]
            if len(glob.glob(os.path.join(pipe,subdir,deriv+'*'))) > 0:
                temp=subprocess.check_output('find '+os.path.join(pipe,subdir,deriv)+' -iname "*.nii.gz" -o -iname "*.csv"', shell=True)

            if len(temp) > 0:
                temp=temp.split('\n')
                temp=[t for t in temp if t != '']
                diffdict=token_set(temp,'/')
                for k1 in diffdict:
                    temp2=diffdict[k1]
                    numlist=[]
                    for t2 in temp2:
                        numlist.append(k1.split('/').index(t2))
                    temp2=[t2 for (nl,t2) in sorted(zip(numlist,temp2))]

                    if len(temp2) > 0:
                        newfile=os.path.join(sldir,pipename,subdirname,deriv,'/'.join(temp2))
                    else:
                        newfile=os.path.join(sldir,pipename,subdirname,deriv,k1.split('/')[-1])
                
                    newdir='/'.join(newfile.split('/')[:-1])
                    if not os.path.isdir(newdir):
                        os.makedirs(newdir)
                    if not os.path.isfile(newfile):
                        #print k1,newfile
                        os.symlink(k1,newfile)
            else:
                print "no files here: ",'find '+os.path.join(pipe,subdir,deriv)+' -iname "*.nii.gz" -o -iname "*.csv"'




"""
for k1 in big_dict.keys():
    for k2 in big_dict[k1].keys():
        #diffdict=token_set(big_dict[k1][k2].split('\n'),'/')
        #for k3 in diffdict.keys():
        #    print k3, diffdict[k3]
        print big_dict[k1][k2].split('\n')



#fo=open('strats.csv', 'w')
for bkey in big_dict.keys():
	for lkey in big_dict[bkey].keys():
		#print "Running for "+bkey+","+lkey
		filelist=sorted(big_dict[bkey][lkey].split('\n'))
		
		filelist=sorted([f.strip() for f in filelist if any(s in f for s in sublist)])
		if 'scrub' in bkey:
			filelist=sorted([f for f in filelist if '/_threshold_0.2/' in f])

		uniques = sorted(set(['/'.join(f.replace(bkey+'/','').split('/')[1:]) for f in filelist]))
		
		
		for unq_num, unqfile in enumerate(uniques):
			#if '.csv' in unqfile:
			#	print unqfile
			#fo.write(bkey+lkey+str(unq_num)+','+unqfile+'\n')
			#print "Strategy "+str(unq_num)

			filelist_filt=[f for f in filelist if unqfile in f]


			if 'scrub' in bkey:
				s='1'
			else:
				s='0'
			if 'global0' in unqfile:
				g='0'
			elif 'global1' in unqfile:
				g='1'
			else:
				g=''
			if 'fnirt' in bkey:
				regtype='fnirt'
			else:
				regtype='ants'

			#opname=bkey.replace(preproc_direc+regdir, '').replace('/','')+'_'+lkey+'_'+str(unq_num)
			alias=name_key.translation[name_key.cpac_op==lkey].values[0]
			fold=name_key.fold[name_key.cpac_op==lkey].values[0]
			if 'centrality' in lkey and 'lfcd' in unqfile.split('/')[-1]:
				add='_'.join(unqfile.split('/')[-1].split('_')[0:2])
			elif 'centrality' in lkey and 'degree' in unqfile.split('/')[-1]:
				add='_'.join(unqfile.split('/')[-1].split('_')[0:3:2])
			elif ('dr_' in lkey) and ('z' not in lkey):
				add=unqfile.split('/')[-1].split('_')[3]
			elif ('dr_' in lkey) and ('z' in lkey):
				add=unqfile.split('/')[-1].split('_')[4]
			else:
				add=''

			if add != '':
				opname=alias+'_'+add
			else:
				opname=alias
			#opname=alias+'_'+regtype+'_s'+s+'_g'+g+'_'+add
			#opname=lkey+'_'+str(unq_num)+'_'+regtype+'_s'+s+'_g'+g+'_'+unqfile.split('/')[-1].split('.')[0]
			#print opname
			#print unqfile.split('/')[1]

			for f in filelist_filt:
				subsesh= f.replace(preproc_direc+regdir, '').split('/')[2]
				sub=subsesh.split('_')[0]
				sesh='_'.join(subsesh.split('_')[1:])
				scan='_'.join(unqfile.split('/')[1].split('_')[2:-1])
				

				if 'anat' in lkey:

					opsldir = os.path.join(sldir, regtype+'_s0_g0', sub+'-'+sesh, fold)
					if not os.path.isdir(opsldir):
						os.makedirs(opsldir)
					if os.path.isfile(opsldir+'/'+opname+'.nii.gz'):
						os.remove(opsldir+'/'+opname+'.nii.gz')
					os.symlink(f, opsldir+'/'+opname+'.nii.gz')

				elif 'roi' in lkey:
					opsldir = os.path.join(sldir, regtype+'_s'+s+'_g'+g, sub+'-'+sesh,scan,fold)
					if not os.path.isdir(opsldir):
						os.makedirs(opsldir)
					if os.path.isfile(opsldir+'/'+unqfile.split('/')[-1]):
						os.remove(opsldir+'/'+unqfile.split('/')[-1])
					os.symlink(f, opsldir+'/'+unqfile.split('/')[-1])

				else:
					opsldir = os.path.join(sldir, regtype+'_s'+s+'_g'+g, sub+'-'+sesh,scan,fold)
					if not os.path.isdir(opsldir):
						os.makedirs(opsldir)
					if os.path.isfile(opsldir+'/'+opname+'.nii.gz'):
						os.remove(opsldir+'/'+opname+'.nii.gz')
					os.symlink(f, opsldir+'/'+opname+'.nii.gz')

#fo.close()
"""
