def analyze_rounds(self):
    n_folds = 10
    round_groups = self.df_all.groupby('round_cat')
    strat_features = ['50-450_slope_cat', '450-950_slope_cat', '950-1450_slope_cat', '1450-1950_slope_cat']
    stroke_features = ['50_stroke_norm', '500_stroke_norm', '1000_stroke_norm', '1500_stroke_norm', '2000_stroke_norm']
    count = 0
    # strokes_rounds = []
    df_count = 0
    for fold in range(n_folds):
        for name, group in round_groups:
            comp_round = group.loc[group.index[0], 'round'][0]
            if comp_round == 'Q':
                continue
            group = group.sample(n=600)
            group_type = 'rounds'
            print('Round: %s' % comp_round)
            name_fig = '../figures/group_diff_plots/rounds/' + str(comp_round)
            strategies = group.loc[:, strat_features].values
            p1 = [item[0] for item in strategies]
            p2 = [item[1] for item in strategies]
            p3 = [item[2] for item in strategies]
            p4 = [item[3] for item in strategies]
            # strokes = group.loc[:,stroke_features].values
            # strat_plotter = sp.strat_plotting(strategies, strokes, name_fig)
            # strat_plotter.plot_strategies()
            # amount, names = self.tactic_graph(group, strat_features, comp_round, group_type)
            mean_strat_round = group.loc[:, strat_features].mean().values
            mean_stroke_round = group.loc[:, stroke_features].mean().values
            # strat_plotter = sp.strat_plotting([mean_strat_round], [mean_stroke_round], name_fig + '_mean')
            # strat_plotter.plot_strategies()
            name_round_list_mean = [comp_round] * 4
            name_round_list_long = [comp_round] * len(strategies)
            dist = [500, 1000, 1500, 2000]
            if count == 0:
                mean_strats_rounds = np.array(mean_strat_round)
                round_column = np.array(name_round_list_mean)
                distances = np.array(dist)
                # amounts = [amount]
                # names_amounts = [names]
                previous_rounds = [comp_round]
                df_all_strats = pd.DataFrame(data={'rounds': name_round_list_long, 'strat_p1': p1, 'strat_p2': p2,
                                                   'strat_p3': p3, 'strat_p4': p4})
            else:
                mean_strats_rounds = np.hstack([mean_strats_rounds, mean_strat_round])
                round_column = np.hstack([round_column, name_round_list_mean])
                distances = np.hstack([distances, dist])
                # self.round_differences(amounts,amount,names_amounts,names, previous_rounds, comp_round)
                # amounts.append(amount)
                # names_amounts.append(names)
                previous_rounds.append(comp_round)
                new_part_df_all_strats = pd.DataFrame(data={'rounds': name_round_list_long, 'strat_p1': p1,
                                                            'strat_p2': p2, 'strat_p3': p3, 'strat_p4': p4})
                df_all_strats = pd.concat([df_all_strats, new_part_df_all_strats])
            # strokes_rounds.append(strokes_rounds)
            count += 1
        # analyze_df_mean_strats = pd.DataFrame(data={'strat':mean_strats_rounds, 'dist':distances, 'rounds':round_column})
        # mm.linear_mixed_model(analyze_df_mean_strats)
        new_results_df = mm.nearest_neighbor_all_parts(df_all_strats, grouping_type=group_type)
        if df_count == 0:
            results_df = new_results_df
        else:
            results_df = pd.concat([results_df, new_results_df])
        df_count += 1
    results_df = results_df.reset_index(drop=True)
    mm.summarize_kfold_results(results_df, group_type)


def round_differences(self, amounts, amount, names_amounts, names, previous_rounds, current_round):
    # if current_round == 'Q':
    # if len(previous_rounds) == 1:
    #     amounts = [amounts]
    #     names_amounts = [names_amounts]
    for round_index, round_amounts in enumerate(amounts):
        names_for_round = names_amounts[round_index]
        diffs_round = []
        names_found = []
        # Create the same x-axis for the round amount and the current amount
        # Make sure that there are 0's in the places where the other has a value and this one not
        #
        x = []
        y = []
        for item_index, item in enumerate(round_amounts):
            name_for_item = names_for_round[item_index]
            for index_current_round_item, name_in_current_round in enumerate(names):
                if name_for_item == name_in_current_round:
                    current_item_amount = amount[index_current_round_item]
                    diffs_round.append(abs(current_item_amount - item))
                    names_found.append(name_for_item)
                    x.append(item)
                    y.append(current_item_amount)
        for item_index, item in enumerate(round_amounts):
            name_for_item = names_for_round[item_index]
            if name_for_item in names_found:
                continue
            else:
                diffs_round.append(item)
                names_found.append(name_for_item)
                x.append(item)
                y.append(0)
        for index_current_round_item, name_in_current_round in enumerate(names):
            if name_in_current_round in names_found:
                continue
            else:
                diffs_round.append(amount[index_current_round_item])
                names_found.append(name_for_item)
                x.append(0)
                y.append(amount[index_current_round_item])
        wilcoxon = ss.wilcoxon(x, y)
        print('wilcoxon: %s' % str(wilcoxon))
        diffs_round_names = names_found
        # print('prev round name: %s' % previous_rounds[round_index])
        # print('this round name: %s' % current_round)
        # print('differences this round %s' % diffs_round)
        # print('names strategies to differences %s' % diffs_round_names)
        # print('mean difference round %s' % np.mean(diffs_round))
        # print('mean amount round %s' % np.mean(amount))
        # print('mean amount previous round %s' % np.mean(round_amounts))
