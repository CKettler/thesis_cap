import pandas as pd
import numpy as np

class time_correction:
    def __init__(self, all_df):
        self.all_df = all_df
        self.time_df = pd.DataFrame()
        self.time_df_part = pd.DataFrame()
        self.speed_df = pd.DataFrame()
        self.speed_df_part = pd.DataFrame()
        self.presence_check = 0

    def create_dfs(self):
        time_cols = [col for col in self.all_df.columns if 'time' in col and '-' not in col]
        speed_cols = [col for col in self.all_df.columns if 'speed' in col and '-' not in col and '2000' not in col]
        self.time_df = self.all_df.loc[:, time_cols]
        self.speed_df = self.all_df.loc[:, speed_cols]


    def define_part(self, part, end_point_speed, section_size):
        if part == 0:
            begin_point_speed = 0
            end_point_speed = begin_point_speed + section_size
        else:
            begin_point_speed = end_point_speed - 1
            end_point_speed = begin_point_speed + (section_size + 1)
        self.speed_df_part = self.speed_df.iloc[:, begin_point_speed:end_point_speed]
        self.time_df_part = self.time_df.iloc[:, part:part+1]
        return end_point_speed

    def calculate_time_with_speed(self, race_nr):
        time_list = []
        current_race_speeds = self.speed_df_part.iloc[race_nr,:]
        current_race_time = self.time_df_part.iloc[race_nr,:]
        for speed_per_50 in current_race_speeds:
            # v = d/t therefore t = d/v
            time_50 = 50.0/float(speed_per_50)
            time_list.append(time_50)
        total_time = np.sum(time_list)
        dif = current_race_time - total_time
        print(dif)

    def partition_and_check(self):
        section_size_speed = len(self.speed_df.columns.values) / 4
        end_point_speed = 0
        for part in range(4):
            end_point_speed = self.define_part(part, end_point_speed, section_size_speed)
            for race_nr in range(len(self.speed_df.index.values)):
                self.calculate_time_with_speed(race_nr)

    def correct_speeds_with_time(self):
        self.create_dfs()
        self.partition_and_check()

