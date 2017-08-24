import pandas as pd
import strat_plotting as sp
import pylab as pl
import matplotlib.pyplot as plt
import mixed_models as mm
import numpy as np
import scipy.stats as ss

class group_analysis:
    def __init__(self, df_all):
        self.df_all = df_all

    def analyze_all_groups(self):
        # self.analyze_groups(catagory_name='weight_cat', string_feature='weight_class', n_folds=1)
        # self.analyze_groups(catagory_name='round_cat', string_feature='round', n_folds=1)
        # self.analyze_groups(catagory_name='year', string_feature='year', n_folds=1)
        self.analyze_groups(catagory_name='contest_cat', string_feature='contest', n_folds=1)
        # self.analyze_groups(catagory_name='avg_opp', string_feature='average_rank_opponents', n_folds=1)
        # self.analyze_groups(catagory_name='boat_cat', string_feature='boatsize', n_folds=1)
        # self.analyze_groups(catagory_name='sex', string_feature='sex', n_folds=1)
        # self.analyze_groups(catagory_name='coxswain', string_feature='coxswain', n_folds=1)
        # self.analyze_groups(catagory_name='races_after_day', string_feature='races_after_day', n_folds=1)
        # self.analyze_groups(catagory_name='countries', string_feature='countries', n_folds=1)
        # self.analyze_groups_teams(catagory_name='team_cat', string_feature='team_cat', n_folds=1)
        # self.analyze_groups(catagory_name='2000m_rank', string_feature='2000m_rank', n_folds=1)
        # self.analyze_groups_ranks(catagory_name='2000m_rank', string_feature='2000m_rank', n_folds=1,
        #                           dependent_features=['boatsize', 'round_cat', 'races_after_day'])

    def analyze_groups(self, catagory_name, string_feature, n_folds):
        if catagory_name == 'round_cat' or catagory_name == 'team_cat':
            groups = self.df_all.groupby(catagory_name)
        if catagory_name == 'avg_opp':
            avg_opp_values = self.df_all.loc[:,string_feature].values
            self.df_all['average_rank_opponents_more_precision'] = avg_opp_values
            rounded_values_list = []
            for value in avg_opp_values:
                # print('value: %s' % value)
                rounded_value = round(value * 2) / 2
                # print('rounded value: %s' % rounded_value)
                rounded_values_list.append(rounded_value)
            self.df_all.loc[:, string_feature] = rounded_values_list
            groups = self.df_all.groupby(string_feature)
        else:
            groups = self.df_all.groupby(string_feature)
            group_sizes = groups.size().values
            if catagory_name == 'team_cat':
                minimum_group_size_considered = 15
            else:
                minimum_group_size_considered = 50
            group_sizes_used = [group_size for group_size in group_sizes if group_size > minimum_group_size_considered]
            smallest_group = np.min(group_sizes_used) - 1
            print('groupsize used for %s is %s' % (string_feature,smallest_group))
            strat_features = ['50-450_slope_cat', '450-950_slope_cat', '950-1450_slope_cat', '1450-1950_slope_cat']
            stroke_features = ['50_stroke_norm', '500_stroke_norm', '1000_stroke_norm', '1500_stroke_norm', '2000_stroke_norm']
            count = 0
            # strokes_rounds = []
            df_count = 0
            for fold in range(n_folds):
                group_count = 0
                for name, group in groups:
                    if catagory_name == 'round_cat':
                        group_name = group.loc[group.index[0], string_feature][0]
                    elif catagory_name == 'team_cat':
                        group_name = str(group.loc[group.index[0], ['countries', 'boattype']].values)
                    else:
                        group_name = group.loc[group.index[0], string_feature]
                    if group.shape[0] < smallest_group:
                        # print(group_name)
                        group_count += 1
                        continue
                    group_count += 1
                    if group_name == 'Q':
                        continue
                    group = group.sample(n=smallest_group)
                    group_type = string_feature
                    print(string_feature + ': %s' % group_name)
                    name_fig = '../figures/group_diff_plots/' + string_feature + '/' + str(group_name)
                    strategies = group.loc[:, strat_features].values
                    p1 = [item[0] for item in strategies]
                    p2 = [item[1] for item in strategies]
                    p3 = [item[2] for item in strategies]
                    p4 = [item[3] for item in strategies]
                    # strokes = group.loc[:,stroke_features].values
                    # strat_plotter = sp.strat_plotting(strategies, strokes, name_fig)
                    # strat_plotter.plot_strategies()
                    # amount, names = self.tactic_graph(group, strat_features, comp_round, group_type)
                    mean_strat_group = group.loc[:, strat_features].mean().values
                    mean_stroke_group = group.loc[:, stroke_features].mean().values
                    # mean_stroke_group = group.loc[:, stroke_features].quantile(q=0.5).values
                    # max_strokes_group = group.loc[:, stroke_features].quantile(q=0.75).values
                    # min_strokes_group = group.loc[:, stroke_features].quantile(q=0.25).values
                    max_strokes_group, min_strokes_group = self.calculate_confidence_interval(group, stroke_features)
                    strat_plotter = sp.strat_plotting([mean_strat_group], [mean_stroke_group], max_strokes_group,
                                                      min_strokes_group, name_fig + '_mean_quant')
                    strat_plotter.plot_strategies()
            #         name_group_list_mean = [group_name] * 4
            #         name_group_list_long = [group_name] * len(strategies)
            #         dist = [500,1000,1500,2000]
            #         if count == 0:
            #             mean_strats_group_names = np.array(mean_strat_group)
            #             group_name_columns = np.array(mean_stroke_group)
            #             distances = np.array(dist)
            #             # amounts = [amount]
            #             # names_amounts = [names]
            #             # previous_rounds = [group_name]
            #             df_all_strats = pd.DataFrame(data={'groups': name_group_list_long, 'strat_p1': p1, 'strat_p2': p2,
            #                                                'strat_p3': p3, 'strat_p4': p4})
            #         else:
            #             mean_strats_group_names = np.hstack([mean_strats_group_names, mean_strat_group])
            #             group_name_columns = np.hstack([group_name_columns, name_group_list_mean])
            #             distances = np.hstack([distances, dist])
            #             # self.round_differences(amounts,amount,names_amounts,names, previous_rounds, comp_round)
            #             # amounts.append(amount)
            #             # names_amounts.append(names)
            #             # previous_rounds.append(group_name)
            #             new_part_df_all_strats = pd.DataFrame(data={'groups': name_group_list_long, 'strat_p1': p1,
            #                                                         'strat_p2': p2, 'strat_p3': p3, 'strat_p4': p4})
            #             df_all_strats = pd.concat([df_all_strats, new_part_df_all_strats])
            #         # strokes_rounds.append(strokes_rounds)
            #         print(count)
            #         count += 1
            #     # analyze_df_mean_strats = pd.DataFrame(data={'strat':mean_strats_rounds, 'dist':distances, 'rounds':round_column})
            #     # mm.linear_mixed_model(analyze_df_mean_strats)
            #     if fold == 0:
            #         df_strat_all_folds = df_all_strats
            #     else:
            #         df_strat_all_folds = pd.concat([df_strat_all_folds, df_all_strats])
            #     # new_results_df = mm.nearest_neighbor_all_parts(df_all_strats, group_type=group_type)
            #     # if df_count == 0:
            #     #     results_df = new_results_df
            #     # else:
            #     #     results_df = pd.concat([results_df, new_results_df])
            #     df_count += 1
            # # results_df = results_df.reset_index(drop=True)
            # mm.histogram_comparison(df_strat_all_folds, group_type, smallest_group, boattype=None, roundtype=None, races_after=None)
            # # mm.summarize_kfold_results(results_df, group_type)

    def calculate_confidence_interval(self, group, column_names):
        max_strokes = []
        min_strokes = []
        for i in range(5):
            feature = column_names[i]
            strokes = group.loc[:, feature].values
            mean, sigma = strokes.mean(), strokes.std(ddof=1)
            if sigma == 0:
                max_strokes.append(mean)
                min_strokes.append(mean)
                continue
            conf_int_a = ss.norm.interval(0.95, loc=mean, scale=sigma)
            max_strokes.append(conf_int_a[1])
            min_strokes.append(conf_int_a[0])
        return max_strokes, min_strokes


    def analyze_groups_teams(self, catagory_name, string_feature, n_folds):
        if catagory_name == 'team_cat':
            groups_per_boat = self.df_all.groupby(['boat_cat'])
            for name_group_boat, group_per_boat in groups_per_boat:
                boattype = group_per_boat.loc[group_per_boat.index[0],'boattype']
                print(boattype)
                if 'LM2x' not in boattype:
                    continue
                groups = group_per_boat.groupby(['country_cat'])
            # if catagory_name == 'round_cat' or catagory_name == 'team_cat':
            #     groups = self.df_all.groupby(catagory_name)
            # if catagory_name == 'avg_opp':
            #     avg_opp_values = self.df_all.loc[:,string_feature].values
            #     self.df_all['average_rank_opponents_more_precision'] = avg_opp_values
            #     rounded_values_list = []
            #     for value in avg_opp_values:
            #         # print('value: %s' % value)
            #         rounded_value = round(value * 2) / 2
            #         # print('rounded value: %s' % rounded_value)
            #         rounded_values_list.append(rounded_value)
            #     self.df_all.loc[:, string_feature] = rounded_values_list
            #     groups = self.df_all.groupby(string_feature)
            # else:
            #     groups = self.df_all.groupby(string_feature)
                group_sizes = groups.size().values
                if catagory_name == 'team_cat':
                    minimum_group_size_considered = 15
                else:
                    minimum_group_size_considered = 50
                group_sizes_used = [group_size for group_size in group_sizes if group_size > minimum_group_size_considered]
                if not group_sizes_used:
                    continue
                smallest_group = np.min(group_sizes_used) - 1
                print('groupsize used for %s is %s' % (string_feature,smallest_group))
                strat_features = ['50-450_slope_cat', '450-950_slope_cat', '950-1450_slope_cat', '1450-1950_slope_cat']
                stroke_features = ['50_stroke_norm', '500_stroke_norm', '1000_stroke_norm', '1500_stroke_norm', '2000_stroke_norm']
                count = 0
                # strokes_rounds = []
                df_count = 0
                for fold in range(n_folds):
                    group_count = 0
                    for name, group in groups:
                        if catagory_name == 'round_cat':
                            group_name = group.loc[group.index[0], string_feature][0]
                        elif catagory_name == 'team_cat':
                            group_name = str(group.loc[group.index[0], ['countries', 'boattype']].values)
                        else:
                            group_name = group.loc[group.index[0], string_feature]
                        if group.shape[0] < smallest_group:
                            # print(group_name)
                            group_count += 1
                            continue
                        group_count += 1
                        if group_name == 'Q':
                            continue
                        group = group.sample(n=smallest_group)
                        group_type = string_feature
                        print(string_feature + ': %s' % group_name)
                        name_fig = '../figures/group_diff_plots/' + string_feature + '/' + str(group_name)
                        strategies = group.loc[:, strat_features].values
                        p1 = [item[0] for item in strategies]
                        p2 = [item[1] for item in strategies]
                        p3 = [item[2] for item in strategies]
                        p4 = [item[3] for item in strategies]
                        # strokes = group.loc[:,stroke_features].values
                        # strat_plotter = sp.strat_plotting(strategies, strokes, name_fig)
                        # strat_plotter.plot_strategies()
                        # amount, names = self.tactic_graph(group, strat_features, comp_round, group_type)
                        mean_strat_group = group.loc[:, strat_features].mean().values
                        mean_stroke_group = group.loc[:, stroke_features].mean().values
                        # max_strokes_group = group.loc[:, stroke_features].quantile(q=0.75).values
                        # min_strokes_group = group.loc[:, stroke_features].quantile(q=0.25).values
                        max_strokes_group, min_strokes_group = self.calculate_confidence_interval(group,
                                                                                                  stroke_features)
                        strat_plotter = sp.strat_plotting([mean_strat_group], [mean_stroke_group], max_strokes_group,
                                                          min_strokes_group, name_fig + '_mean_quant')
                        strat_plotter.plot_strategies()
                #         name_group_list_mean = [group_name] * 4
                #         name_group_list_long = [group_name] * len(strategies)
                #         dist = [500,1000,1500,2000]
                #         if count == 0:
                #             mean_strats_group_names = np.array(mean_strat_group)
                #             group_name_columns = np.array(mean_stroke_group)
                #             distances = np.array(dist)
                #             # amounts = [amount]
                #             # names_amounts = [names]
                #             # previous_rounds = [group_name]
                #             df_all_strats = pd.DataFrame(data={'groups': name_group_list_long, 'strat_p1': p1, 'strat_p2': p2,
                #                                                'strat_p3': p3, 'strat_p4': p4})
                #         else:
                #             mean_strats_group_names = np.hstack([mean_strats_group_names, mean_strat_group])
                #             group_name_columns = np.hstack([group_name_columns, name_group_list_mean])
                #             distances = np.hstack([distances, dist])
                #             # self.round_differences(amounts,amount,names_amounts,names, previous_rounds, comp_round)
                #             # amounts.append(amount)
                #             # names_amounts.append(names)
                #             # previous_rounds.append(group_name)
                #             new_part_df_all_strats = pd.DataFrame(data={'groups': name_group_list_long, 'strat_p1': p1,
                #                                                         'strat_p2': p2, 'strat_p3': p3, 'strat_p4': p4})
                #             df_all_strats = pd.concat([df_all_strats, new_part_df_all_strats])
                #         # strokes_rounds.append(strokes_rounds)
                #         print(count)
                #         count += 1
                #     # analyze_df_mean_strats = pd.DataFrame(data={'strat':mean_strats_rounds, 'dist':distances, 'rounds':round_column})
                #     # mm.linear_mixed_model(analyze_df_mean_strats)
                #     if fold == 0:
                #         df_strat_all_folds = df_all_strats
                #     else:
                #         df_strat_all_folds = pd.concat([df_strat_all_folds, df_all_strats])
                #     # new_results_df = mm.nearest_neighbor_all_parts(df_all_strats, group_type=group_type)
                #     # if df_count == 0:
                #     #     results_df = new_results_df
                #     # else:
                #     #     results_df = pd.concat([results_df, new_results_df])
                #     df_count += 1
                # # results_df = results_df.reset_index(drop=True)
                # mm.histogram_comparison(df_strat_all_folds, group_type, smallest_group, boattype, roundtype=None, races_after=None)
                # # mm.summarize_kfold_results(results_df, group_type)

    def analyze_groups_ranks(self, catagory_name, string_feature, n_folds, dependent_features):
        groups_per_dependent_features = self.df_all.groupby(dependent_features)
        for name_group_depfeat, group_per_depfeat in groups_per_dependent_features:
            boattype = group_per_depfeat.loc[group_per_depfeat.index[0],'boatsize']
            # if 8 != boattype:
            #     continue
            roundtype = group_per_depfeat.loc[group_per_depfeat.index[0],'round'][0]
            races_after = group_per_depfeat.loc[group_per_depfeat.index[0],'races_after_day']
            print('boatsize: %s' % boattype)
            print('roundtype: %s' % roundtype)
            print('no races after day: %s' % races_after)
            groups = group_per_depfeat.groupby([string_feature])
            group_sizes = groups.size().values
            minimum_group_size_considered = 25
            group_sizes_used = [group_size for group_size in group_sizes if group_size > minimum_group_size_considered]
            if not group_sizes_used:
                continue
            smallest_group = np.min(group_sizes_used) - 1
            print('groupsize used for %s is %s' % (string_feature,smallest_group))
            strat_features = ['50-450_slope_cat', '450-950_slope_cat', '950-1450_slope_cat', '1450-1950_slope_cat']
            stroke_features = ['50_stroke_norm', '500_stroke_norm', '1000_stroke_norm', '1500_stroke_norm', '2000_stroke_norm']
            count = 0
            # strokes_rounds = []
            df_count = 0
            for fold in range(n_folds):
                group_count = 0
                for name, group in groups:
                    group_name = group.loc[group.index[0], string_feature]
                    if group.shape[0] < smallest_group:
                        # print(group_name)
                        group_count += 1
                        continue
                    group_count += 1
                    if group_name == 'Q':
                        continue
                    group = group.sample(n=smallest_group)
                    group_type = string_feature
                    print(string_feature + ': %s' % group_name)
                    name_fig = '../figures/group_diff_plots/' + string_feature + '/' + str(group_name) + '_' + \
                               str(roundtype) + '_' + str(boattype)
                    strategies = group.loc[:, strat_features].values
                    p1 = [item[0] for item in strategies]
                    p2 = [item[1] for item in strategies]
                    p3 = [item[2] for item in strategies]
                    p4 = [item[3] for item in strategies]
                    # strokes = group.loc[:,stroke_features].values
                    # strat_plotter = sp.strat_plotting(strategies, strokes, name_fig)
                    # strat_plotter.plot_strategies()
                    # amount, names = self.tactic_graph(group, strat_features, comp_round, group_type)
                    mean_strat_group = group.loc[:, strat_features].mean().values
                    mean_stroke_group = group.loc[:, stroke_features].mean().values
                    max_strokes_group, min_strokes_group = self.calculate_confidence_interval(group,
                                                                                              stroke_features)
                    strat_plotter = sp.strat_plotting([mean_strat_group], [mean_stroke_group], max_strokes_group,
                                                      min_strokes_group, name_fig + '_mean_quant')
                    strat_plotter.plot_strategies()
            #         name_group_list_mean = [group_name] * 4
            #         name_group_list_long = [group_name] * len(strategies)
            #         dist = [500,1000,1500,2000]
            #         if count == 0:
            #             mean_strats_group_names = np.array(mean_strat_group)
            #             group_name_columns = np.array(mean_stroke_group)
            #             distances = np.array(dist)
            #             # amounts = [amount]
            #             # names_amounts = [names]
            #             # previous_rounds = [group_name]
            #             df_all_strats = pd.DataFrame(data={'groups': name_group_list_long, 'strat_p1': p1, 'strat_p2': p2,
            #                                                'strat_p3': p3, 'strat_p4': p4})
            #         else:
            #             mean_strats_group_names = np.hstack([mean_strats_group_names, mean_strat_group])
            #             group_name_columns = np.hstack([group_name_columns, name_group_list_mean])
            #             distances = np.hstack([distances, dist])
            #             # self.round_differences(amounts,amount,names_amounts,names, previous_rounds, comp_round)
            #             # amounts.append(amount)
            #             # names_amounts.append(names)
            #             # previous_rounds.append(group_name)
            #             new_part_df_all_strats = pd.DataFrame(data={'groups': name_group_list_long, 'strat_p1': p1,
            #                                                         'strat_p2': p2, 'strat_p3': p3, 'strat_p4': p4})
            #             df_all_strats = pd.concat([df_all_strats, new_part_df_all_strats])
            #         # strokes_rounds.append(strokes_rounds)
            #         count += 1
            #     if fold == 0:
            #         df_strat_all_folds = df_all_strats
            #     else:
            #         df_strat_all_folds = pd.concat([df_strat_all_folds, df_all_strats])
            #     df_count += 1
            # mm.histogram_comparison(df_strat_all_folds, group_type, smallest_group, boattype, roundtype, races_after)



    def tactic_graph(self, input_group, features, group_name, group_type):
        tactic_groups = input_group.groupby(features)
        names = []
        strokes = []
        amount = []
        highest_amount = 0
        highest_name = []
        highest_stroke = []
        all_amount = 0
        boundary = 0
        for name, group in tactic_groups:
            if group.shape[0] > boundary:
                all_amount += group.shape[0]
                # if you want to use strategies as names
                # names.append(str(group.loc[group.index[0],'strat_cat']))
                # if you want to use the actual strategies as names
                name = [round(x,1) for x in name]
                name = [0.0 if x == -0.0 else x for x in name]
                if group.shape[0] > 10:
                    names.append(str(name))
                    amount.append(group.shape[0])
                mean_strokes_group = group.loc[:, ['50_stroke_norm', '500_stroke_norm', '1000_stroke_norm',
                                              '1500_stroke_norm', '2000_stroke_norm']].mean()
                strokes.append(mean_strokes_group.values)
                if group.shape[0] > highest_amount:
                    second_highest_amount = highest_amount
                    highest_amount = group.shape[0]
                    second_highest_name = highest_name
                    second_highest_stroke = highest_stroke
                    highest_name = name
                    highest_stroke = mean_strokes_group
        print('most used strategy count: %s' % highest_amount)
        print('most used strategy: %s' % highest_name)
        print('second most used strategy count: %s' % second_highest_amount)
        print('all count: %s' % all_amount)
        print('number of strategies: %s' % len(amount))
        # self.make_bar_graph(amount, names, "tactic_amounts_" + str(boundary) + "plus")
        self.make_bar_graph(amount, names, group_type, group_name + "_tactic_amounts_" + str(boundary) + "plus")
        pl.figure()
        pl.hist(amount, bins=100)  # use this to draw histogram of your data
        pl.title('Histogram of the number of occurences of strategies')
        pl.xlabel('groupsize strategy')
        pl.ylabel('Number of occurences')
        pl.savefig(group_name + 'tactic_amounts_hist.png')
        return amount, names

    def make_bar_graph(self, amounts, names, group_type, figure_name):
        amounts = pd.Series.from_array(amounts)
        # now to plot the figure...
        plt.figure(figsize=(12, 21))
        ax = amounts.plot(kind='bar')
        ax.set_title("Strategy Frequency")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Frequency")
        ax.set_xticklabels(names)
        # ax.set_xticklabels([])
        plt.savefig("../figures/group_diff_plots/" + group_type + "/" + figure_name + ".png")
