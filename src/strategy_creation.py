import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import strat_plotting as str_plt
import pylab as pl
import sklearn.cluster as cl
import clustering as clust

class strategy_creator:
    def __init__(self, df_all):
        self.df_all = df_all
        self.new_all_df = pd.DataFrame()
        self.speed_stroke_prop_pointer = False

    def divide_in_strategies(self):
        # country_dict = self.create_counts_countries()
        # boattype_dict, boat_cat_to_boattype = self.create_counts_boattypes()
        self.df_all = self.df_all.round({'firstlast_50_stroke-450_stroke':0, 'firstlast_450_stroke-950_stroke':0,
                           'firstlast_950_stroke-1450_stroke':0, 'firstlast_1450_stroke-1950_stroke':0})
        new_feat_names = self.create_strategies()
        self.create_strokes()
        strokes = self.df_all.loc[:,['50_stroke_norm', '500_stroke_norm', '1000_stroke_norm', '1500_stroke_norm',
                              '2000_stroke_norm']]
        strokes = strokes.values
        #Strategy statistics
        # tactic_groups = self.df_all.groupby(new_feat_names)
        # This is to be able to identify the different strategies
        # self.categorize_strategies(tactic_groups)
        # tactic_groups = self.df_all.groupby(new_feat_names)
        # self.strategy_proportions(tactic_groups, whole_race_values_per_box=None)
        strategies = self.gradient_plot(new_feat_names)
        # tactic_groups = self.df_all.groupby(['team_cat'])
        # highest_name, second_highest_name, strategies, strokes, boundary = self.tactic_graph(tactic_groups)
        figure_name = '../figures/strategies/allstrategies'
        strat_plotter = str_plt.strat_plotting(strategies, strokes, figure_name)
        strat_plotter.plot_strategies()
        self.var_strat_team()
        if self.speed_stroke_prop_pointer:
            self.speed_stoke_prop_plot()

        return self.df_all

    def create_strategies(self):
        col_list = ['firstlast_50_stroke-450_stroke', 'firstlast_450_stroke-950_stroke',
                    'firstlast_950_stroke-1450_stroke', 'firstlast_1450_stroke-1950_stroke']
        new_feat_names = ['50-450_slope_cat', '450-950_slope_cat', '950-1450_slope_cat', '1450-1950_slope_cat']
        self.df_all = self.df_all.dropna(subset=['firstlast_50_stroke-450_stroke', 'firstlast_450_stroke-950_stroke',
                                                 'firstlast_950_stroke-1450_stroke',
                                                 'firstlast_1450_stroke-1950_stroke'])
        self.df_all = self.df_all.reset_index(drop=True)
        for i, col in enumerate(col_list):
            new_feat_list = []
            values = []
            for index, diff in enumerate(self.df_all.loc[:, col]):
                # new_feat_list.append(value)
                average_stroke = self.df_all.loc[index, 'average_stroke_pace']
                if diff == 0 or diff == 0.0:
                    value = 0.0
                else:
                    value = diff / average_stroke
                    # value = round(value, 1)
                values.append(value)
                new_feat_list.append(value)
            print('max: %s' % np.max(new_feat_list))
            print('min: %s' % np.min(new_feat_list))
            self.df_all[new_feat_names[i]] = new_feat_list
        return new_feat_names

    def gradient_plot(self, new_feat_names):
        gradients_matrix = self.df_all.loc[:,new_feat_names].values
        names = ['50-500m', '500-1000m', '1000-1500m', '1500-2000m']
        for race_part in range(0, 4):
            print(race_part)
            gradient = [item[race_part] for item in gradients_matrix if item[race_part] > -0.6]
            race_part_name = names[race_part]
            pl.figure()
            pl.hist(gradient, bins=100)  # use this to draw histogram of your data
            pl.title('Distribution of gradients of race part %s' % race_part_name)
            pl.xlabel('gradient')
            pl.ylabel('frequency')
            pl.savefig('../figures/gradient_hists/histogram_gradients_' + race_part_name + '.png')
            print('mean ' + race_part_name + ' : %s' % np.mean(gradient))
            print('std ' + race_part_name + ' : %s' % np.std(gradient))
        return gradients_matrix

    def create_strokes(self):
        col_stroke_vals = ['50_stroke', '500_stroke', '1000_stroke', '1500_stroke', '2000_stroke']
        new_feat_true_vals = ['50_stroke_norm', '500_stroke_norm', '1000_stroke_norm', '1500_stroke_norm',
                              '2000_stroke_norm']
        self.df_all = self.df_all.dropna(
            subset=['50_stroke', '500_stroke', '1000_stroke', '1500_stroke', '2000_stroke'])
        self.df_all = self.df_all.reset_index(drop=True)
        for j, col2 in enumerate(col_stroke_vals):
            new_feat_list = []
            values = []
            for index2, stroke in enumerate(self.df_all.loc[:, col2]):
                # new_feat_list.append(value)
                average_stroke = self.df_all.loc[index2, 'average_stroke_pace']
                if stroke == 0 or stroke == 0.0:
                    value = 0
                else:
                    value = stroke / average_stroke
                    value = round(value, 1)
                values.append(value)
                new_feat_list.append(value)
            self.df_all[new_feat_true_vals[j]] = new_feat_list

    def make_bar_graph(self, amounts, names, figure_name):
        amounts = pd.Series.from_array(amounts)
        # now to plot the figure...
        plt.figure(figsize=(12, 21))
        ax = amounts.plot(kind='bar')
        ax.set_title("Strategy Frequency")
        ax.set_xlabel("Strategy")
        ax.set_ylabel("Frequency")
        ax.set_xticklabels(names)
        # ax.set_xticklabels([])
        plt.savefig("../figures/" + figure_name + ".png")

    def tactic_graph(self, tactic_groups):
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
                # name = [round(x,1) for x in name]
                # name = [0.0 if x == -0.0 else x for x in name]
                # names.append(name)
                amount.append(group.shape[0])
                mean_strokes_group = group.loc[:, ['50_stroke_norm', '500_stroke_norm', '1000_stroke_norm',
                                              '1500_stroke_norm', '2000_stroke_norm']].mean()
                strokes.append(mean_strokes_group.values)
                if group.shape[0] > highest_amount:
                    second_highest_amount = highest_amount
                    highest_amount = group.shape[0]
                    second_highest_name = highest_name
                    highest_name = name
                    highest_stroke = mean_strokes_group
        print('most used strategy count: %s' % highest_amount)
        print('most used strategy: %s' % highest_name)
        # print('second most used strategy count: %s' % second_highest_amount)
        print('all count: %s' % all_amount)
        print('number of strategies: %s' % len(amount))
        # self.make_bar_graph(amount, names, "tactic_amounts_" + str(boundary) + "plus")
        self.make_bar_graph(amount, names, "tactic_amounts_" + str(boundary) + "plus")
        pl.figure()
        pl.hist(amount, bins=100)  # use this to draw histogram of your data
        pl.title('Histogram of the number of occurences of strategies')
        pl.xlabel('groupsize strategy')
        # pl.title('Histogram of the number of occurences of the different group sizes of the teams')
        # pl.xlabel('group size team')
        pl.ylabel('Number of occurrences')
        pl.savefig('../figures/team_occurrence_histogram.png')
        # pl.savefig('../figures/tactic_amounts_hist.png')
        return highest_name, second_highest_name, names, strokes, boundary


    def tactic_graph_country(self, groups_strat_country, country_name):
        country_strategy_amount = []
        country_strategy_names = []
        for name, group in groups_strat_country:
            if country_name in str(name):
                country_strategy_names.append(name[:4])
                country_strategy_amount.append(group.shape[0])
        self.make_bar_graph(country_strategy_amount, country_strategy_names, country_name+"_strategy_plot")


    def create_counts_countries(self):
        country_groups = self.df_all.groupby(self.df_all.countries.str[:3])
        country_dict = {}
        for name, group in country_groups:
            country_dict[name] = group.shape[0]
        return country_dict

    def create_counts_boattypes(self):
        boattype_groups = self.df_all.groupby(self.df_all.boat_cat)
        boattype_dict = {}
        boat_cat_to_boattype = {}
        for name, group in boattype_groups:
            boattype_dict[name] = group.shape[0]
            boattypes = group.loc[:,'boattype'].unique()
            for boattype in boattypes:
                boat_cat_to_boattype[boattype] = name
        return boattype_dict, boat_cat_to_boattype

    def one_tactic_country(self, groups_strat_country, country_dict):
        one_tactic_countries = []
        one_tactic_countries_tactics = []
        one_tactic_countries_amount = []
        for name, group in groups_strat_country:
            for country in country_dict:
                if country in name:
                    current_percentage = float(group.shape[0]) / float(country_dict[country])
                    if current_percentage > 0.5:
                        one_tactic_countries.append(country)
                        one_tactic_countries_tactics.append(name)
                        one_tactic_countries_amount.append(group.shape[0])
        for index, one_tactic_country in enumerate(one_tactic_countries):
            print('one tactic country: %s' % one_tactic_country)
            print('country tactic: %s' % str(one_tactic_countries_tactics[index]))
            print('amount in country: %s' % one_tactic_countries_amount[index])

    def highest_percentage_country(self, groups_strat_country, country_dict):
        highest_percentage = 0.0
        for name, group in groups_strat_country:
            for country in country_dict:
                if country in name:
                    current_percentage = float(group.shape[0]) / float(country_dict[country])
                    if current_percentage > highest_percentage:
                        highest_country = country
                        country_amount = country_dict[country]
                        highest_percentage = current_percentage
                        highest_name = name
                        amount_in_group = group.shape[0]
        print('highest country: %s' % highest_country)
        print("highest name: %s" % str(highest_name))
        print("amount in group: %s" % amount_in_group)
        print("amount in country: %s" % country_amount)
        print("percentage: %s" % highest_percentage)
        print('--------' * 20)

    def one_tactic_catbased(self, groups_strat, dict, entity_type):
        one_tactic_entity = []
        one_tactic_entity_tactics = []
        one_tactic_entity_amount = []
        for name, group in groups_strat:
            for key in dict:
                if key == name[4]:
                    current_percentage = float(group.shape[0]) / float(dict[key])
                    if current_percentage > 0.5:
                        one_tactic_entity.append(group.loc[0, entity_type])
                        one_tactic_entity_tactics.append(name)
                        one_tactic_entity_amount.append(group.shape[0])
        for index, one_tactic_country in enumerate(one_tactic_entity):
            print("one tactic " + str(entity_type) + ": %s" % one_tactic_country)
            print( str(entity_type) + " tactic: %s" % str(one_tactic_entity_tactics[index]))
            print("amount in " + str(entity_type) + ": %s" % one_tactic_entity_amount[index])

    def highest_percentage_catbased(self, groups_strat, dict, entity_type):
        highest_percentage = 0.0
        for name, group in groups_strat:
            group = group.reset_index()
            for key in dict:
                if key == name[4]:
                    current_percentage = float(group.shape[0]) / float(dict[key])
                    if current_percentage > highest_percentage:
                        highest_entity = group.loc[0, entity_type]
                        entity_amount = dict[key]
                        highest_percentage = current_percentage
                        highest_name = name
                        amount_in_group = group.shape[0]
        print("highest " + str(entity_type) + ": %s" % highest_entity)
        print("highest name: %s" % str(highest_name))
        print("amount in group: %s" % amount_in_group)
        print("amount in " + str(entity_type) + ": %s" % entity_amount)
        print("percentage: %s" % highest_percentage)
        print('--------' * 20)

    def tactic_graph_catbased(self, groups_strat, cat_to_type_dict, entity_name, plot_entity):
        strategy_amount = []
        strategy_names = []
        for name, group in groups_strat:
            group_entity_instances = group.loc[:, entity_name].unique()
            if plot_entity in group_entity_instances or 'H' + plot_entity in group_entity_instances:
                for instance in group_entity_instances:
                    catagory = cat_to_type_dict[instance]
        for name, group in groups_strat:
            if catagory == name[4]:
                strategy_names.append(name[:4])
                strategy_amount.append(group.shape[0])
        self.make_bar_graph(strategy_amount, strategy_names, entity_name + "_" + plot_entity + "_strategy_plot")

    def categorize_strategies(self, strategy_groups):
        category = 0
        largest_group_size = 0
        largest_group_cat = 0
        print('[giving each strategy a category...]')
        for name, group in strategy_groups:
            category_list = [category] * group.shape[0]
            group['strat_cat'] = category_list
            if category == 0:
                new_all_df = group
            else:
                new_all_df = new_all_df.append(group)
            if group.shape[0] > largest_group_size:
                largest_group_size = group.shape[0]
                largest_group_cat = category
            category += 1
        print('number of different strategies: %s' %category)
        print('category belonging to most used strategy: %s' %largest_group_cat)
        sort_cols = ['year', 'contest_cat', 'round_cat', 'round_number', 'boat_cat', 'start_lane']
        self.df_all = new_all_df.sort_values(by=sort_cols)
        self.df_all = self.df_all.reset_index()


    def strategy_proportions(self, tactic_groups, whole_race_values_per_box):
        #Not correct yet, should be the differences check the stacking etc
        count = 0
        for name, group in tactic_groups:
            name = [name[0], name[1], name[2], name[3]]
            # strat_plotter = str_plt.strat_plotting(whole_race_values_per_box, name, plot_indicator=False)
            # box_lower_line, box_upper_line = strat_plotter.plot_strategies()
            # both_lines = np.vstack([box_upper_line, box_lower_line])
            # mean_line = np.mean(both_lines, axis=0)
            mean_line = name
            list1 = [0,0,1,2]
            list2 = [1,3,2,3]
            props = []
            for index, j in enumerate(list1):
                k = list2[index]
                if mean_line[j] == 0:
                    if mean_line[k] == 0:
                        prop = 1
                    else:
                        prop = mean_line[k]
                elif mean_line[k] == 0:
                    prop = mean_line[j]
                else:
                    prop = mean_line[j]/mean_line[k]
                props.append(prop)
            group['1_2_prop'] = [props[0]] * group.shape[0]
            group['1_4_prop'] = [props[1]] * group.shape[0]
            group['2_3_prop'] = [props[2]] * group.shape[0]
            group['3_4_prop'] = [props[3]] * group.shape[0]
            self.create_dataframe_from_groups(group, count)
            count += 1
        self.df_all = self.new_all_df


    def create_dataframe_from_groups(self, team, count):
        if count == 0:
            self.new_all_df = team
        else:
            self.new_all_df = pd.concat([self.new_all_df, team])


    def var_strat_team(self):
        teams = self.df_all.groupby(['countries', 'boat_cat'])
        used_cols = ['50-450_slope_cat', '450-950_slope_cat', '950-1450_slope_cat', '1450-1950_slope_cat']
        count = 0
        for name, team in teams:
            strat_features = team.loc[:, used_cols].as_matrix()
            team_variance = np.var(strat_features, axis=0)
            team['strat_part_1_var'] = [team_variance[0]] * team.shape[0]
            team['strat_part_2_var'] = [team_variance[1]] * team.shape[0]
            team['strat_part_3_var'] = [team_variance[2]] * team.shape[0]
            team['strat_part_4_var'] = [team_variance[3]] * team.shape[0]
            self.create_dataframe_from_groups(team, count)
            count += 1
        self.df_all = self.new_all_df

    def clustering(self, new_feat_names, nr_clusters):
        self.df_all = self.df_all.dropna(subset=new_feat_names)
        new_feat_df = self.df_all.loc[:, new_feat_names]
        x_matrix = new_feat_df.as_matrix()
        # kmeans = cl.KMeans(n_clusters=nr_clusters).fit(x_matrix)
        # cluster_centers = kmeans.cluster_centers_
        clusterer = clust.clustering(self.df_all, new_feat_names)
        cluster_centers, self.df_all = clusterer.dtw_clustering()
        x = np.arange(4)
        for cluster_center in cluster_centers:
            plt.plot(x, cluster_center)
        plt.show()
        # self.df_all['labels'] = kmeans.labels_
        cluster_groups = self.df_all.groupby(['label'])
        count = 0
        highest_group_len = 0
        highest_group = None
        highest_group_name = 0
        for name, group in cluster_groups:
            cluster_center_list = [cluster_centers[int(name)]] * group.shape[0]
            # group.loc[:, new_feat_names] = cluster_center_list
            # self.create_dataframe_from_groups(group, count)
            if group.shape[0] > highest_group_len:
                if count > 0:
                    second_highest_group_len = highest_group_len
                    second_highest_group = highest_group
                    second_highest_group_name = highest_group_name
                    second_highest_group_index = highest_group_index
                highest_group_len = group.shape[0]
                highest_group = group
                highest_group_name = cluster_centers[int(name)]
                highest_group_index = name
            count += 1

        group_strats = highest_group.loc[:, new_feat_names].as_matrix()
        print('highest group name: %s' % str(highest_group_name))
        print('highest group len: %s' % str(highest_group_len))
        print('highest group index: %s' % str(highest_group_index))
        for strat in group_strats:
            plt.plot(x, strat)
        # plt.plot(x, highest_group_name)
        plt.show()
        # self.df_all = self.new_all_df

    def speed_stoke_prop_plot(self):
        speed_stroke_list = self.df_all.loc[:, 'speed/stroke'].values
        speed_stroke_list = np.array([x for x in speed_stroke_list if str(x) != 'nan' and x != 0.0])
        highest_prop = np.max(speed_stroke_list)
        print('highest speed/stroke prop: %s' % highest_prop)
        lowest_prop = np.min(speed_stroke_list)
        print('lowest speed/stroke prop: %s' % lowest_prop)
        print('mean: %s' % np.mean(speed_stroke_list))
        print('std: %s' % np.std(speed_stroke_list))
        pl.figure()
        pl.hist(speed_stroke_list, bins=100)  # use this to draw histogram of your data
        pl.title('Histogram of the number of occurences of the effectsize of one stroke')
        pl.xlabel('Effect (speed(m/s)/strokes(sps))')
        pl.ylabel('Number of occurences')
        pl.savefig('../figures/speed_stroke_prop_hist.png')

