import glob
import pandas as pd
import numpy as np

all_csvs = glob.glob('../results/ranks/*.csv')
for csv in all_csvs:
    print(csv)
    significance_count = 0
    df = pd.read_csv(csv)
    p_values = df.loc[:,'uneq_var_ttest_p'].values
    p_values = [0.5 if p_value == 1.0 else p_value for p_value in p_values]
    t_values = df.loc[:,'uneq_var_ttest_t'].values
    most_significant = np.argmin(p_values)
    least_significant = np.argmax(p_values)
    if isinstance(most_significant, list):
        index_most_significant = np.argmax(abs(t_values[most_significant]))
    else:
        index_most_significant = most_significant
    if isinstance(least_significant, list):
        index_least_significant = np.argmin(abs(t_values[least_significant]))
    else:
        index_least_significant = least_significant
    for p in p_values:
        if p < 0.05:
            significance_count += 1
    print('most significant: %s' % df.loc[index_most_significant, :])
    print('least significant: %s' % df.loc[index_least_significant, :])
    print('number of significant rows: %s' % significance_count)
