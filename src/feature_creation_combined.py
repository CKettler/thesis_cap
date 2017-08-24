import pandas as pd
import numpy as np
import math

#turn off chained warnings
pd.options.mode.chained_assignment = None

class feature_creation_combined:
    def __init__(self, dataframe_all):
        self.df_all = dataframe_all
        self.speed_part = []
        self.time_part = []
        self.boat_speeds = []
        self.boat_times = []
        self.new_all_df = pd.DataFrame

    def create_all_features(self):
        self.dif_median()
        self.coxswain_feature()
        self.same_day_feature()
        self.opposing_teams()
        self.total_strokes_feature()
        return self.df_all

    def define_part(self, part, end_point_speed, section_size):
        """
        Make two dataframes for a part of the race. One with the speeds for that part en an other with the times of that part
        :param part: part number
        :param end_point_speed: The index of the last end point
        :param section_size: the size of the section
        :return: end_point_speed. It will be necessary when creating the next part
        """
        if part == 0:
            begin_point_speed = 0
            end_point_speed = begin_point_speed + section_size
        else:
            begin_point_speed = end_point_speed
            end_point_speed = begin_point_speed + section_size
        self.speed_part = self.boat_speeds[begin_point_speed:end_point_speed]
        self.time_part = self.boat_times[part:part+1]
        return end_point_speed

    def average_speed_500_scale(self):
        """
        Get the average speed over the 500 meters and scale the speeds
        :return: speed_scale_per_500, which is a list of 4 lists, each list containing the scaling of all speeds in 500m.
        And time_per_500, which is the time clocked on this 500m
        """
        section_size_speed_list = [10, 10, 10, 9]
        end_point_speed = 0
        speed_scale_per_500 = []
        time_per_500 = []
        for part in range(4):
            end_point_speed = self.define_part(part, end_point_speed, section_size_speed_list[part])
            average_speed = np.mean(self.speed_part)
            speed_diff = (self.speed_part - average_speed)
            speed_part = speed_diff/average_speed
            speed_scale = 1+(speed_part)
            speed_scale_per_500.append(speed_scale)
            time_per_500 += self.time_part
        return speed_scale_per_500, time_per_500


    def scale_times(self, speed_scale_per_500, time_per_500):
        """
        scale the times per 50m using the speed scales
        :param speed_scale_per_500: a list of lists containing the scale of the speeds of each 50m compared to the
        average speed of each 500m
        :return: a list of lists containing the times of each 50m
        """
        part_size = [10, 10, 10, 9]
        previous_time = 0
        for part in range(4):
            # time_per_500 - previous_time to calculate the time the boat used to cover 500m
            between_time = time_per_500[part]-previous_time
            time_per_50_list = [between_time/10] * part_size[part]
            if part == 0:
                scaled_times_500 = np.multiply(time_per_50_list, speed_scale_per_500[part])
                scaled_times = scaled_times_500
            elif part == 3:
                scaled_times_500 = np.multiply(time_per_50_list, speed_scale_per_500[part])
                dif = between_time - np.sum(scaled_times_500)
                scaled_times_500 = np.append(scaled_times_500,[dif])
                scaled_times = np.append(scaled_times, scaled_times_500)
            else:
                scaled_times_500 = np.multiply(time_per_50_list, speed_scale_per_500[part])
                scaled_times = np.append(scaled_times, scaled_times_500)
            dif = np.sum(scaled_times_500) - between_time
            previous_time = time_per_500[part]
        end_dif = time_per_500[3] - np.sum(scaled_times)
        return scaled_times



    def time_per_50_m(self, race):
        """
        Use the speed and the time to get a time for each 50 meters for each boat
        :return: all_boats_scaled_times, which is a list of x lists, where x is the number of boats participating in the
         race. Each list contains the times derived from the speeds and time per 50m
        """
        race = race.reset_index()
        all_boats_scaled_times = []
        # Get an average speed. Check the scale of each speed referenced from the average speed.
        speed_cols = [col for col in self.df_all.columns if 'speed' in col and '-' not in col and '2000' not in col]
        time_cols = [col for col in self.df_all.columns if 'time' in col and '-' not in col]
        for boat_ix in range(int(race.shape[0])):
            self.boat_speeds = race.loc[boat_ix, speed_cols].values.tolist()
            self.boat_times = race.loc[boat_ix, time_cols].values.tolist()
            speed_scale_per_500, time_per_500 = self.average_speed_500_scale()
            # Spread the time accordingly tot the speeds scales
            scaled_times_boat = self.scale_times(speed_scale_per_500, time_per_500)
            all_boats_scaled_times.append(scaled_times_boat)
        return all_boats_scaled_times

    def time_per_50_to_df(self, time_50_lists, race_index):
        new_time_50_lists = []
        for list in time_50_lists:
            new_race_list = []
            for index in range(len(list)):
                new_race_list.append(sum(list[:index+1]))
            new_time_50_lists.append(new_race_list)
        time_per_50_df = pd.DataFrame(new_time_50_lists)
        time_per_50_df.columns = [str(x * 50) + 'm-time_approx' for x in range(1,time_per_50_df.shape[1]+1)]
        time_per_50_df.index = race_index
        return time_per_50_df

    def rank_per_50_df(self, time_per_50_df):
        columns = [str(x * 50) + 'm-rank_approx' for x in range(1, time_per_50_df.shape[1] + 1)]
        rank_per_50_df = pd.DataFrame(index=time_per_50_df.index, columns=columns)
        rank_per_50_df = rank_per_50_df.fillna(0)
        for index in range(0, time_per_50_df.shape[1]):
            times_per_distance = time_per_50_df.iloc[:,index].tolist()
            sorted_index = sorted(range(len(times_per_distance)), key=lambda k: times_per_distance[k])
            ranks = [x + 1 for x in sorted_index]
            rank_per_50_df.iloc[:,index] = ranks
        return rank_per_50_df

    def get_median(self, race):
        """
        Get median race. In this case a self created race that has the ending time in the middle of the middle two boats
        concerning ending time or the middle boat if it is an odd number
        :param race: a df of the current race
        :return:
        """
        finish_times = race.loc[:, '2000m_time'].values.tolist()
        # If it is an even number, the median is between the middle two finish times
        if len(finish_times)%2==0:
            mid_low_rank = len(finish_times) / 2
            mid_high_rank = mid_low_rank - 1
            mid_slow_time = finish_times[mid_low_rank]
            mid_fast_time = finish_times[mid_high_rank]
            median_race_2000m_time = np.mean([mid_fast_time, mid_slow_time])
        # If it is an odd number, the median is the middle finish time
        else:
            mid_rank = int(math.ceil(len(finish_times) / 2.0) - 1)
            median_race_2000m_time = finish_times[mid_rank]
        full_median_boat_race = [median_race_2000m_time/40.0] * 40
        return full_median_boat_race

    def compute_dif_races(self, median_boat, real_boats):
        """
        Computes the difference between the median boat and the real boats from the race
        :param median_race: boat that goes in even pace to the median 2000m time
        :param real_race: time per 50m of the boats that are in the race
        :return: a list of x lists containing the difference between the median race and the boats. In which x is the
        number of boats in the race
        """
        all_boats_dif = []
        for boat in real_boats:
            dif_boat_med = boat - median_boat
            all_boats_dif.append(dif_boat_med)
        return all_boats_dif

    def create_dataframe_from_groups(self, race, count):
        if count == 0:
            self.new_all_df = race
        else:
            self.new_all_df = pd.concat([self.new_all_df, race])

    def dif_median(self):
        """
        Calculate the difference in time between the reference crew
        and the current team. Need the time per 50 meters ti calculate
        this
        :return:
        """
        # The columns that identify a race. If grouped by these columns each race is seperated in a different group
        race_id_columns = ['year', 'contest_cat', 'boat_cat', 'round_cat', 'round_number']
        races = self.df_all.groupby(race_id_columns)
        col_names_new = ['time_dif_median_' + str((x+1)*50) for x in range(40)]
        count = 0
        for name, race in races:
            median_boat_race = self.get_median(race)
            time_per_50_all_boats = self.time_per_50_m(race)
            times_50_df = self.time_per_50_to_df(time_per_50_all_boats, race_index=race.index)
            race = race.join(times_50_df)
            ranks_50_df = self.rank_per_50_df(times_50_df)
            race = race.join(ranks_50_df)
            all_boats_dif = self.compute_dif_races(median_boat_race, time_per_50_all_boats)
            all_boats_dif_df = pd.DataFrame(all_boats_dif)
            all_boats_dif_df.columns = col_names_new
            all_boats_dif_df = all_boats_dif_df.set_index(race.index.values)
            race = race.join(all_boats_dif_df)
            self.create_dataframe_from_groups(race, count)
            count += 1
        self.df_all = self.new_all_df

    def contains_plus(self, row):
        if '+' in row['boattype']:
            return 1
        else:
            return 0

    def coxswain_feature(self):
        """
        A feature stating whether there is a coxswain present in the boat or not
        :return: Feature is added to self.df_all
        """
        self.df_all['coxswain'] = self.df_all.apply(lambda row: self.contains_plus(row), axis=1)

    def opposing_teams(self):
        """
        The teams opposing the current team
        :return: 6 columns are created containing the country ID's of the opponents
        """
        races_grouped = self.df_all.groupby(['year', 'contest_cat', 'round_cat', 'round_number', 'boat_cat'])
        opponent1_list = []
        opponent2_list = []
        opponent3_list = []
        opponent4_list = []
        opponent5_list = []
        opponent1_rank_list = []
        opponent2_rank_list = []
        opponent3_rank_list = []
        opponent4_rank_list = []
        opponent5_rank_list = []
        opponent1_stroke_list = []
        opponent2_stroke_list = []
        opponent3_stroke_list = []
        opponent4_stroke_list = []
        opponent5_stroke_list = []
        opponent1_speed_list = []
        opponent2_speed_list = []
        opponent3_speed_list = []
        opponent4_speed_list = []
        opponent5_speed_list = []
        opponent_average_rank_list = []
        opponent_average_stroke_list = []
        opponent_average_speed_list = []
        count = 0
        for name, race in races_grouped:
            race = race.reset_index(drop=True)
            if count == 0:
                new_df = race
                count += 1
            else:
                new_df = new_df.append(race)
                count += 1
            for boat_index in range(race.shape[0]):
                opposing_list = []
                opposing_rank_list = []
                opposing_stroke_list = []
                opposing_speed_list = []
                for opposing_index in range(6):
                    if boat_index == opposing_index:
                        continue
                    elif opposing_index >= race.shape[0]:
                        opposing_list.append(-1)
                        opposing_rank_list.append(-1)
                        opposing_stroke_list.append(-1)
                        opposing_speed_list.append(-1)
                    else:
                        opposing_list.append(race.loc[opposing_index, 'country_cat'])
                        opposing_rank_list.append(race.loc[opposing_index, 'average_rank_team'])
                        opposing_stroke_list.append(race.loc[opposing_index, 'average_stroke_team'])
                        opposing_speed_list.append(race.loc[opposing_index, 'average_speed_team'])
                if not opposing_rank_list:
                    mean_opponents_rank = 0
                else:
                    mean_opponents_rank = np.mean([x for x in opposing_rank_list if x != -1])
                if not opposing_stroke_list:
                    mean_opponents_stroke = 0
                else:
                    mean_opponents_stroke = np.mean([x for x in opposing_stroke_list if x != -1])
                if not opposing_speed_list:
                    mean_opponents_speed = 0
                else:
                    mean_opponents_speed = np.mean([x for x in opposing_speed_list if x != -1])
                opponent_average_rank_list.append(mean_opponents_rank)
                opponent_average_stroke_list.append(mean_opponents_stroke)
                opponent_average_speed_list.append(mean_opponents_speed)
                opponent1_list.append(opposing_list[0])
                opponent2_list.append(opposing_list[1])
                opponent3_list.append(opposing_list[2])
                opponent4_list.append(opposing_list[3])
                opponent5_list.append(opposing_list[4])
                opponent1_rank_list.append(opposing_rank_list[0])
                opponent2_rank_list.append(opposing_rank_list[1])
                opponent3_rank_list.append(opposing_rank_list[2])
                opponent4_rank_list.append(opposing_rank_list[3])
                opponent5_rank_list.append(opposing_rank_list[4])
                opponent1_stroke_list.append(opposing_stroke_list[0])
                opponent2_stroke_list.append(opposing_stroke_list[1])
                opponent3_stroke_list.append(opposing_stroke_list[2])
                opponent4_stroke_list.append(opposing_stroke_list[3])
                opponent5_stroke_list.append(opposing_stroke_list[4])
                opponent1_speed_list.append(opposing_speed_list[0])
                opponent2_speed_list.append(opposing_speed_list[1])
                opponent3_speed_list.append(opposing_speed_list[2])
                opponent4_speed_list.append(opposing_speed_list[3])
                opponent5_speed_list.append(opposing_speed_list[4])
        self.df_all = new_df.reset_index(drop=True)
        self.df_all['opponent1'] = opponent1_list
        self.df_all['opponent2'] = opponent2_list
        self.df_all['opponent3'] = opponent3_list
        self.df_all['opponent4'] = opponent4_list
        self.df_all['opponent5'] = opponent5_list
        self.df_all['opponent_rank1'] = opponent1_rank_list
        self.df_all['opponent_rank2'] = opponent2_rank_list
        self.df_all['opponent_rank3'] = opponent3_rank_list
        self.df_all['opponent_rank4'] = opponent4_rank_list
        self.df_all['opponent_rank5'] = opponent5_rank_list
        self.df_all['opponent_avg_stroke1'] = opponent1_stroke_list
        self.df_all['opponent_avg_stroke2'] = opponent2_stroke_list
        self.df_all['opponent_avg_stroke3'] = opponent3_stroke_list
        self.df_all['opponent_avg_stroke4'] = opponent4_stroke_list
        self.df_all['opponent_avg_stroke5'] = opponent5_stroke_list
        self.df_all['opponent_avg_speed1'] = opponent1_speed_list
        self.df_all['opponent_avg_speed2'] = opponent2_speed_list
        self.df_all['opponent_avg_speed3'] = opponent3_speed_list
        self.df_all['opponent_avg_speed4'] = opponent4_speed_list
        self.df_all['opponent_avg_speed5'] = opponent5_speed_list
        self.df_all['average_rank_opponents'] = opponent_average_rank_list
        self.df_all['average_stroke_opponents'] = opponent_average_stroke_list
        self.df_all['average_speed_opponents'] = opponent_average_speed_list


    def number_races_day_count(self, races):
        number_races_day = len(races.loc[:, 'round'].values.tolist())
        if number_races_day == 2:
            another_race_after = [1, 0]
            another_race_before = [0, 1]
            if 'R' in races['round'][races.index[1]]:
                after_is_rep = [1, 0]
                before_is_rep = [0, 0]
            elif 'R' in races['round'][races.index[0]]:
                after_is_rep = [0, 0]
                before_is_rep = [0, 1]
            else:
                after_is_rep = [0, 0]
                before_is_rep = [0, 0]
        elif number_races_day == 3:
            another_race_after = [2, 1, 0]
            another_race_before = [0, 1, 2]
            if 'R' in races['round'][races.index[1]]:
                after_is_rep = [1, 0, 0]
                before_is_rep = [0, 0, 1]
            else:
                after_is_rep = [0, 0, 0]
                before_is_rep = [0, 0, 0]
        else:
            another_race_after = [0]
            another_race_before = [0]
            after_is_rep = [0]
            before_is_rep = [0]
        races['races_after_day'] = another_race_after
        races['races_before_day'] = another_race_before
        races['after_is_rep'] = after_is_rep
        races['before_is_rep'] = before_is_rep
        return races

    def same_day_feature(self):
        team_day_cols = ['year', 'contest_cat', 'boat_cat', 'date', 'countries']
        races_team_day = self.df_all.groupby(team_day_cols)
        count = 0
        for name, races in races_team_day:
            races = self.number_races_day_count(races)
            self.create_dataframe_from_groups(races, count)
            count += 1
        self.df_all = self.new_all_df

    def rounds_to_go_feature(self):
        team_comp_cols = ['year', 'contest_cat', 'boat_cat', 'countries']
        races_team_comp = self.df_all.groupby(team_comp_cols)
        count = 0
        for name, races in races_team_comp:
            number_races_comp = len(races.loc[:, 'round'].values.tolist())

    def total_strokes_feature(self):
        stroke_cols = [str(col) for x, col in enumerate(self.df_all.columns) if 'stroke' in col and len(str(col)) < 20]
        nr_rows = self.df_all.shape[0]
        self.df_all['total_stroke_count'] = [0] * nr_rows
        for row in range(nr_rows):
            strokes_array_mean = self.df_all.loc[row,stroke_cols].values.mean()
            strokes_mean_per_second = strokes_array_mean/60.0
            final_time = self.df_all.loc[row,'2000m_time']
            total_stroke_count = final_time * strokes_mean_per_second
            self.df_all.loc[row,'total_stroke_count'] = total_stroke_count






