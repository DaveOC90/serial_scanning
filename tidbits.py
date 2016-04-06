def cpac_motion_gather():
    motdict={}
    for mf in motionfiles:
        motdict[mf]=pd.read_csv(mf,index_col=False)
    allmot=pd.concat(motdict.values())

