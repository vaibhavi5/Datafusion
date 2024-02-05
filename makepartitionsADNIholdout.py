import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/data/users2/vitkyal/projects/SMLvsDL/multicomplex.csv')  # 505 vs 316

tr_smp_sizes = [505]
va_size = 75
te_size = 150
va_te_size = va_size + te_size
nReps = 20

sdir = './SampleSplits_3wayholdoutstratify'

try:
    os.stat(sdir)
except:
    os.mkdir(sdir)

# Hold-out test subjects
df_remaining, df_te = train_test_split(df, test_size=te_size, shuffle=True, stratify=df['ResearchGroup'])  # Stratified split

for tss in tr_smp_sizes:
    for rep in np.arange(nReps):
        # Shuffle order of the test subjects within the remaining subjects
        df_te_shuffled = df_te.sample(frac=1)
        
        # Split remaining subjects into training and validation, preserving class distribution
        df_tr, df_va = train_test_split(df_remaining, test_size=va_size, shuffle=True, stratify=df_remaining['ResearchGroup'])
        
        df_tr.to_csv(sdir+'/tr_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_va.to_csv(sdir+'/va_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_te_shuffled.to_csv(sdir+'/te_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
