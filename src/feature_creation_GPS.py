import pandas as pd
import numpy as np
import exp_smoothing as smooth

class feature_creation_gps:
    def __init__(self, dataframe_all, parts):
        self.df_all = dataframe_all
        self.parts = parts
        self.temp_df = pd.DataFrame()
        self.change_pointer = 0
        self.new_all_df = pd.DataFrame()

    def create_all_features(self):
        """
        Creates all features for both the speeds and the strokes that can be made with just the speeds or just the strokes
        :return: The enriched dataframe with all data. New features can be found at the end of the dataframe
        """
        self.create_boatsize_sex_weight_feature()
        self.create_team_feature()
        self.smooth_strokes()
        self.create_dif_feature()
        self.create_sprint_feature()
        self.create_average_strokepace_feature()
        self.create_average_sprint_feature()

        return self.df_all


    def create_dif_feature(self):
        """
        Create two features. One that describes the average slope over a part of the race and an other that describes
        the difference between the first and the last value of the part
        :return:
        """
        # Get all columns that are stroke data and create a dataframe with all stroke data
        stroke_cols = [col for col in self.df_all.columns if 'stroke' in col and '2000' not in col]
        stroke_df = self.df_all.loc[:, stroke_cols]
        self.divide_in_race_phases(stroke_df)

        # Get all columns that are speeds data and create a dataframe with all speeds data
        speed_cols = [col for col in self.df_all.columns if 'speed' in col and '2000' not in col]
        speed_df = self.df_all.loc[:, speed_cols]
        self.divide_in_race_phases(speed_df)


    def divide_in_race_phases(self, df):
        # nr columns used in slicing
        col_names = df.columns.values.tolist()
        nr_of_columns = len(col_names)
        # section size is determined by the number of parts decided on when calling the function
        section_size = nr_of_columns/self.parts

        for i in range(self.parts):
            # begin point of the slice
            if i == 0:
                begin_point = 0
                end_point = begin_point + section_size
            else:
                begin_point = end_point - 1
                end_point = begin_point + (section_size+2)

            # end point of the slice

            df_section = df.iloc[:,begin_point:end_point]
            # df containing the differences between the cells vertically
            difference_df = df_section.diff(axis=1)
            # overal_diff contains the average difference between the cells in the current section
            overal_diff = difference_df.mean(axis=1)
            # print(df_section)
            section_col_names = df_section.columns.values.tolist()
            # first last diff contains the difference between the first value and last value of the section per row
            first_last_diff = df_section[[str(section_col_names[0]),
                                          str(section_col_names[len(section_col_names)-1])]].diff(axis=1)


            self.df_all.insert(len(self.df_all.columns), 'firstlast_' + str(section_col_names[0]) + '-' +
                           str(section_col_names[len(section_col_names)-1]),
                           first_last_diff[str(section_col_names[len(section_col_names)-1])])
            self.df_all.insert(len(self.df_all.columns), 'avg_' + str(section_col_names[0]) + '-' +
                           str(section_col_names[len(section_col_names) - 1]), overal_diff)

        return df

    def filter_irregularities(self, diffs_list, index_list):
        self.change_pointer = 0
        for index, diff in enumerate(diffs_list[:-1]):
            if index in index_list:
                diff_between_diffs_next = diff + diffs_list[index+1]
                if not diff_between_diffs_next:
                    diffs_list[index] = 0.0
                    diffs_list[index+1] = 0.0
                    self.change_pointer = 1
                elif index > 0:
                    diff_between_diffs_prev = diffs_list[index-1] + diff
                    if not diff_between_diffs_prev:
                        diffs_list[index-1] = 0.0
                        diffs_list[index] = 0.0
                        self.change_pointer = 1
        return diffs_list

    def determine_sprint_point(self, difference_list_strokes):
        index_list_strokes = [index for index, difference in enumerate(difference_list_strokes) if difference > 1.5]
        if not index_list_strokes:
            sprint_point = 0
        else:
            difference_list_strokes = self.filter_irregularities(difference_list_strokes, index_list_strokes)
            if self.change_pointer == 1:
                index_list_strokes = [index for index, difference in enumerate(difference_list_strokes) if difference > 1.5]
            if not index_list_strokes:
                sprint_point = 0
            elif len(index_list_strokes) == 1:
                sprint_point = index_list_strokes[0]
            else:
                max_diffs = [difference_list_strokes[index] for index in index_list_strokes]
                biggest_diff = max_diffs.index(max(max_diffs))
                sprint_point = index_list_strokes[biggest_diff]
        return sprint_point

    def create_sprint_feature(self):
        sprint_point_all = []
        stroke_columns = [col for col in self.df_all.columns if 'stroke' in col and '-' not in col and '2000' not in col]
        stroke_columns_1500_to_2000 = stroke_columns[-10:]
        stroke_values = self.df_all[stroke_columns_1500_to_2000]
        no_datapoints = stroke_values.iloc[:,1].shape[0]
        for boat_index in range(no_datapoints):
            boat_strokes = stroke_values.iloc[boat_index,:].values.tolist()
            difference_list_strokes = [boat_strokes[index+1] - stroke for index, stroke in enumerate(boat_strokes[:-1])]
            sprint_point = self.determine_sprint_point(difference_list_strokes)
            if sprint_point == 0:
                sprint_dist = -1
            else:
                sprint_col = stroke_columns_1500_to_2000[sprint_point + 1]
                sprint_dist = int(sprint_col[:4])
            sprint_point_all.append(sprint_dist)
        self.df_all.insert(len(self.df_all.columns), 'start_sprint', sprint_point_all)

    def create_average_strokepace_feature(self):
        types = ['stroke', 'speed']
        for averaging_type in types:
            used_cols = [str(col) for x, col in enumerate(self.df_all.columns) if averaging_type in col and len(str(col)) < 20]
            teams = self.df_all.groupby(['countries', 'boat_cat'])
            count = 0
            for name, team in teams:
                # if count > 1507:
                average_type_team = team.loc[:, used_cols].mean(axis=1)
                num_averages = len(average_type_team)
                team_average = np.mean(average_type_team)
                team_variance = np.var(average_type_team)
                average_rank_feat = [team_average] * num_averages
                variance_rank_feat = [team_variance] * num_averages
                if averaging_type == 'stroke':
                    team['average_stroke_pace'] = average_type_team
                    team['average_stroke_pace_per_second'] = average_type_team/60
                if averaging_type == 'speed':
                    team['average_speed'] = average_type_team
                team['average_' + averaging_type + '_team'] = average_rank_feat
                team['variance_' + averaging_type + '_team'] = variance_rank_feat
                self.create_dataframe_from_groups(team, count)
                count += 1
            self.df_all = self.new_all_df
        self.df_all['speed/stroke'] = np.where(self.df_all['average_stroke_pace_per_second'] < 1, self.df_all['average_stroke_pace_per_second'],
                                               self.df_all['average_speed'] / self.df_all['average_stroke_pace_per_second'])
        # print('average stroke pace all: %s' % self.df_all.loc[:, 'average_stroke_team'].mean())

    def create_average_sprint_feature(self):
        types = ['start_sprint']
        for averaging_type in types:
            teams = self.df_all.groupby(['countries', 'boat_cat'])
            count = 0
            for name, team in teams:
                sprint_team = team.loc[:, 'start_sprint']
                sprint_boolean_team = [1.0 if sprint > 0.0 else 0.0 for sprint in sprint_team]
                num_bools = len(sprint_boolean_team)
                team_average = np.mean(sprint_boolean_team)
                team_variance = np.var(sprint_boolean_team)
                average_sprint_feat = [team_average] * num_bools
                variance_sprint_feat = [team_variance] * num_bools
                team['sprint_bool'] = sprint_boolean_team
                team['average_' + 'sprint' + '_team'] = average_sprint_feat
                team['variance_' + 'sprint' + '_team'] = variance_sprint_feat
                self.create_dataframe_from_groups(team, count)
                count += 1
            self.df_all = self.new_all_df
            # print('average stroke pace all: %s' % self.df_all.loc[:, 'average_stroke_team'].mean())

    def create_boatsize_sex_weight_feature(self):
        count = 0
        groups = self.df_all.groupby(['boattype'])
        for name, group in groups:
            boattype = group.loc[group.index[0], 'boattype']
            if 'M' in boattype:
                group['sex'] = 'M'
                group['sex_cat'] = 1
            else:
                group['sex'] = 'W'
                group['sex_cat'] = 0
            if 'L' in boattype:
                group['weight_class'] = 'Light'
                group['weight_cat'] = 0
            else:
                group['weight_class'] = 'Heavy'
                group['weight_cat'] = 1
            size = [int(s) for s in list(name) if s.isdigit()]
            group['boatsize'] = size[0]
            self.create_dataframe_from_groups(group, count)
            count += 1
        self.df_all = self.new_all_df

    def create_team_feature(self):
        count = 0
        groups = self.df_all.groupby(['boattype', 'countries'])
        for name, group in groups:
            group['team_cat'] = count
            self.create_dataframe_from_groups(group, count)
            count += 1
        self.df_all = self.new_all_df

    def create_dataframe_from_groups(self, team, count):
        if count == 0:
            self.new_all_df = team
        else:
            self.new_all_df = pd.concat([self.new_all_df, team])

    def smooth_strokes(self):
        nr_rows = self.df_all.shape[0]
        col_names = [str(50*x) + '_stroke' for x in range(1,41)]
        name_col_names = ['year', 'countries', 'contest', 'round', 'boattype']
        for i in range(nr_rows):
            strokes_list = self.df_all.loc[i,col_names]
            name_list = self.df_all.loc[i,name_col_names].values
            name_list[0] = str(name_list[0])
            name = '_'.join(name_list)
            smoothed_strokes = smooth.smooth_plot_strokes_list(strokes_list, name, plot_indicator=False)
            self.df_all.loc[i, col_names] = smoothed_strokes



