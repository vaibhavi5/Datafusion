import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

##### MMSE #####
# Multimodal contains 792 total subjects (MCI, Dementia, CN) -- 594, 99, 99
# Binary contains 506 total subjects (Dementia and CN) - 380, 63, 63
df = pd.read_csv('./bincomplex.csv')  # Contains 1051 Subjects - 751

tr_smp_sizes = [350]
va_size = 58
te_size = 58
va_te_size = va_size + te_size
nReps = 20

sdir = './SampleSplits_2way_uniq'

try:
    os.stat(sdir)
except:
    os.mkdir(sdir)

test_subjects_used = []  # Track the test subjects used in previous repetitions

for tss in tr_smp_sizes:
    for rep in np.arange(nReps):
        # Split the data into train and validation+test sets
        df_tr, df_va_te = train_test_split(df, train_size=tss, test_size=va_te_size, shuffle='True')

        # Split the validation+test set into validation and test sets
        df_va, df_te = train_test_split(df_va_te, test_size=te_size, shuffle='True')

        # Filter the test set to ensure unique test subjects for each repetition
        df_te_unique = df_te[~df_te['SubID'].isin(test_subjects_used)].copy()
        test_subjects_used.extend(df_te_unique['SubID'])  # Add the used test subjects to the list

        # Append the remaining test subjects from previous repetitions if necessary
        if len(df_te_unique) < te_size:
            remaining_te_size = te_size - len(df_te_unique)
            remaining_te_subjects = np.random.choice(test_subjects_used, size=remaining_te_size, replace=False)
            df_te_remaining = df_te[df_te['SubID'].isin(remaining_te_subjects)].copy()
            df_te_unique = pd.concat([df_te_unique, df_te_remaining], ignore_index=True)

        # Save the train, validation, and test splits as CSV files
        df_tr.to_csv(sdir + '/tr_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_va.to_csv(sdir + '/va_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
        df_te_unique.to_csv(sdir + '/te_' + str(tss) + '_rep_' + str(rep) + '.csv', index=False)
