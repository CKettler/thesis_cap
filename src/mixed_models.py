import statsmodels.formula.api as smf
import scipy.stats as stats
import numpy as np
import pandas as pd
import pylab as pl
import matplotlib.pyplot as plt

def linear_mixed_model(data):
    md = smf.mixedlm("strat ~ dist", data, groups=data['rounds'])
    mdf = md.fit()
    print(mdf.summary())

def absolute_error(array, value):
    idx = (np.abs(array-value)).argmin()
    ae = np.abs((array[idx]-value))
    array = np.delete(array, idx)
    return ae, array

def nearest_neighbor_all_parts(df_all_strats, group_type):
    groups = df_all_strats.groupby(['groups'])
    count = 0
    df_count = 0
    for name1, group1 in groups:
        for name2, group2 in list(groups)[count+1:]:
            print('comparing group: %s' % name1)
            print('to group: %s' % name2)
            absolute_error_all_parts = []
            for i in range(1, 5):
                absolute_error_all = []
                part_list_adjust = group1.loc[:, 'strat_p' + str(i)].values
                part_list = part_list_adjust
                comparing_part_list = group2.loc[:, 'strat_p' + str(i)].values
                for value in comparing_part_list:
                    ae, part_list_adjust = absolute_error(part_list_adjust, value)
                    absolute_error_all.append(ae)
                mean_absolute_error = np.mean(absolute_error_all)
                absolute_error_all_parts.append(mean_absolute_error)
                var_distance = np.var(absolute_error_all)
                var_values = np.var(df_all_strats.loc[:, 'strat_p' + str(i)])
                # print('part: %s' % i)
                # print('mean distance: %s' % mean_distance)
                # print('variation distance: %s' % var_distance)
                # print('mean round1: %s ' % np.mean(part_list))
                # print('mean round2: %s ' % np.mean(comparing_part_list))
                # print('std round1: %s ' % np.std(part_list))
                # print('std round2: %s ' % np.std(comparing_part_list))
                # print('variation values: %s' % var_values)
                if df_count == 0:
                    results_df = pd.DataFrame(data={'mean_absolute_error': [mean_absolute_error], 'var_dist': [var_distance],
                                               'mean_group1': [np.mean(part_list)],
                                               'mean_group2': [np.mean(comparing_part_list)],
                                               'std_group1': [np.std(part_list)],
                                               'std_group2': [np.std(comparing_part_list)],
                                               'variance_values': [var_values], 'group1': [str(name1)],
                                               'group2': [str(name2)], 'part': [i]})
                else:
                    new_results_df = pd.DataFrame(data={'mean_absolute_error': [mean_absolute_error], 'var_dist': [var_distance],
                                               'mean_group1': [np.mean(part_list)],
                                               'mean_group2': [np.mean(comparing_part_list)],
                                               'std_group1': [np.std(part_list)],
                                               'std_group2': [np.std(comparing_part_list)],
                                               'variance_values': [var_values], 'group1': [str(name1)],
                                               'group2': [str(name2)], 'part': [i]})
                    results_df = pd.concat([results_df,new_results_df])
                df_count += 1
            print('mean distance all parts: %s' % np.mean(absolute_error_all_parts))
        count += 1
    return results_df


def get_unique_names(results_df):
    unique_group1_names = results_df.group1.unique()
    unique_group2_names = results_df.group2.unique()
    all_names_list = np.concatenate([unique_group1_names, unique_group2_names])
    return list(set(all_names_list))

def calculate_q_value(mean_results_part):
    mae = mean_results_part.loc[0,'mean_absolute_error']
    mean_group1 = mean_results_part.loc[0,'mean_group1']
    mean_group2 = mean_results_part.loc[0,'mean_group2']
    q_value = (mean_group1-mean_group2)/mae
    return q_value

def summarize_kfold_results(results_df, group_type):
    unique_group_names = get_unique_names(results_df)
    count1 = 0
    for name1 in unique_group_names:
        for name2 in unique_group_names[count1:]:
            count2 = 0
            for i in range(1, 5):
                part_groups_combi_dataframe = results_df.loc[(results_df['group1'] == name1) &
                                                             (results_df['group2'] == name2) &
                                                             (results_df['part'] == i)]
                if part_groups_combi_dataframe.empty:
                    part_groups_combi_dataframe = results_df.loc[(results_df['group2'] == name1) &
                                                                 (results_df['group1'] == name2) &
                                                                 (results_df['part'] == i)]
                mean_results_part = part_groups_combi_dataframe.mean().to_frame().transpose()
                mean_results_part['group1'] = [name1]
                mean_results_part['group2'] = [name2]
                prop_mean = mean_results_part['mean_absolute_error'].values/mean_results_part['mean_group1'].values
                mean_results_part['mean/mae'] = prop_mean
                if count1+count2 == 0:
                    new_df = mean_results_part
                else:
                    new_df = pd.concat([new_df, mean_results_part])
                count2 += 1
        count1 += 1
    new_df.to_csv('../results/pairwise_comparison_' + str(group_type) + '.csv')
    return new_df


def histogram_comparison(df_strats_groups, group_type, group_size, boattype, roundtype, races_after):
    groups = df_strats_groups.groupby(['groups'])
    count = 0
    df_count = 0
    t_list = []
    p_list = []
    for name1, group1 in groups:
        for name2, group2 in list(groups):
            print('hist comparing group: %s' % name1)
            print('to hist group: %s' % name2)
            for i in range(1, 5):
                list_group1 = group1.loc[:, 'strat_p' + str(i)].values
                list_group2 = group2.loc[:, 'strat_p' + str(i)].values
                if boattype and roundtype:
                    if not races_after is None:
                        comparing_histogram_plot(list_group1, list_group2, name1, name2, group_type, part=i,
                                                 boattype=boattype, roundtype=roundtype, races_after=races_after)
                    else:
                        comparing_histogram_plot(list_group1, list_group2, name1, name2, group_type, part=i,
                                             boattype=boattype, roundtype=roundtype, races_after=None)
                else:
                    comparing_histogram_plot(list_group1, list_group2, name1, name2, group_type, part=i, boattype=None,
                                             roundtype=None, races_after=None)
                t2, p2 = statistical_t_test(list_group1, list_group2, name1, name2, group_type, part=i)
                if df_count == 0:
                    df_ttest = pd.DataFrame(data={'group1':[name1], 'group2':[name2], 'part': [i],
                                                  'uneq_var_ttest_t':[round(t2, 3)], 'uneq_var_ttest_p':[round(p2, 3)]})
                else:
                    new_df_ttest = pd.DataFrame(data={'group1':[name1], 'group2':[name2], 'part': [i],
                                                  'uneq_var_ttest_t':[round(t2, 3)], 'uneq_var_ttest_p':[round(p2, 3)]})
                    df_ttest = pd.concat([df_ttest, new_df_ttest])
                df_count += 1
                t_list.append(abs(t2))
                p_list.append(p2)
            df_ttest = df_ttest[['group1', 'group2', 'part', 'uneq_var_ttest_t', 'uneq_var_ttest_p']]
            # df_ttest.to_csv('../results/ttest_' + str(group_type) + '.csv')
        count += 1
    df_avg = pd.DataFrame(data={'avg_uneq_var_ttest_t':[np.mean(t_list)], 'avg_uneq_var_ttest_p':[np.mean(p_list)]})
    df_ttest = df_ttest[['group1','group2','part','uneq_var_ttest_t','uneq_var_ttest_p']]
    if boattype:
        if roundtype:
            if not races_after is None:
                df_ttest.to_csv('../results/ranks/ttest_' + str(group_type) + '_' + str(boattype) + '_' + str(roundtype)
                                + '_' + str(races_after) + '.csv')
            else:
                df_ttest.to_csv('../results/ranks/ttest_' + str(group_type) + '_' + str(boattype) + '_' + str(roundtype)
                            + '.csv')
        else:
            df_ttest.to_csv('../results/teams/ttest_' + str(group_type) + '_' + str(boattype) + '.csv')
    else:
        df_ttest.to_csv('../results/ttest_' + str(group_type) + '.csv')
    df_avg.to_csv('../results/ttest_avg_' + str(group_type) + '.csv')


def comparing_histogram_plot(list_group1, list_group2, group1_name, group2_name, group_type, part, boattype, roundtype, races_after):
    if part == 1:
        part_name = '50-500'
    elif part == 2:
        part_name = '500-1000'
    elif part == 3:
        part_name = '1000-1500'
    else:
        part_name = '1500-2000'
    pl.figure()
    bins = np.linspace(-0.7, 0.7, 100)
    pl.hist(list_group1, bins=bins, alpha=0.6, label=str(group1_name))
    pl.hist(list_group2, bins=bins, alpha=0.6, label=str(group2_name))
    pl.legend(loc='upper right')
    pl.title('Histograms of the gradient at ' + part_name + ' of groups: ' + str(group1_name) + ' ' + str(group2_name))
    pl.xlabel('gradient')
    pl.ylabel('gradient occurrence')
    if boattype and roundtype:
        if not races_after is None:
            pl.savefig('../figures/group_diff_plots/' + group_type + '/roundboat/' + str(group1_name) + '_' + str(
                group2_name) + '_' + part_name + '_' + str(boattype) + '_' + str(roundtype) + '_' + str(races_after) +
                '_gradient_hist.png')
        else:
            pl.savefig('../figures/group_diff_plots/' + group_type + '/roundboat/' + str(group1_name) + '_' + str(group2_name) +
                   '_' + part_name + '_' + str(boattype) + '_' + str(roundtype) + '_gradient_hist.png')
    else:
        pl.savefig('../figures/group_diff_plots/' + group_type + '/' + str(group1_name) + '_' + str(group2_name) + '_'
                   + part_name + '_gradient_hist.png')


def statistical_t_test(list_group1, list_group2, group1_name, group2_name, group_type, part):
    print('t-test group: %s and %s' % (group1_name, group2_name))
    print('part: %s' % part)
    t2, p2 = stats.ttest_ind(list_group1, list_group2, equal_var=False)
    return t2, p2
