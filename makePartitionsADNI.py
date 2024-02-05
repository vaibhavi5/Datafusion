import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

##### MMSE #####
#Multimodal contains 792 total subjects (MCI, Dementia, CN) -- 594, 99, 99
#Binary contains 506 total subjects (Dementia and CN) - 380, 63, 63
df = pd.read_csv('./mainbinaryfile.csv') # Contains 1051 Subjects - 751

tr_smp_sizes = [356]
va_size = 50
te_size = 100
va_te_size = va_size + te_size
nReps = 20

sdir = './SampleSplits_2complex'

try:
    os.stat(sdir)
except:
    os.mkdir(sdir)  

for tss in tr_smp_sizes:
    for rep in np.arange(nReps):
        df_tr, df_te = train_test_split(df, train_size = tss, test_size=va_te_size, shuffle='True')
        df_va, df_te = train_test_split(df_te, test_size=te_size, shuffle='True')
        df_tr.to_csv(sdir+'/tr_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_va.to_csv(sdir+'/va_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_te.to_csv(sdir+'/te_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
