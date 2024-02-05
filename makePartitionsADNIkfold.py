import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

##### MMSE #####
#Multimodal contains 792 total subjects (MCI, Dementia, CN) -- 594, 99, 99
#Binary contains 506 total subjects (Dementia and CN) - 380, 63, 63
df = pd.read_csv('./bincomplex.csv') # Contains 1051 Subjects - 751

tr_smp_sizes = [350]
va_size = 58
te_size = 58
va_te_size = va_size + te_size
nReps = 20
n_splits = 10  # Number of folds for cross-validation

sdir = './SampleSplits_2way_kfold'

try:
    os.stat(sdir)
except:
    os.mkdir(sdir)

for tss in tr_smp_sizes:
    for rep in np.arange(nReps):
        # Perform k-fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True)

        for fold_idx, (train_index, val_index) in enumerate(kf.split(df)):
            df_train = df.iloc[train_index]
            df_val = df.iloc[val_index]

            # Exclude validation subjects from the remaining pool of subjects
            remaining_subjects = df.index.difference(val_index)
            df_remaining = df.loc[remaining_subjects]

            # Split remaining subjects into test set
            df_test = df_remaining.sample(n=te_size, random_state=rep)

            # Save the splits
            df_train.to_csv(sdir+'/tr_' + str(tss) + '_rep_' + str(rep) + '_fold_' + str(fold_idx) + '.csv', index=False)
            df_val.to_csv(sdir+'/va_' + str(tss) + '_rep_' + str(rep) + '_fold_' + str(fold_idx) + '.csv', index=False)
            df_test.to_csv(sdir+'/te_' + str(tss) + '_rep_' + str(rep) + '_fold_' + str(fold_idx) + '.csv', index=False)


