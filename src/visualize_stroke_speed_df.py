import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from datetime import datetime

class visualize:
    def __init__(self, df, data_type=None, visualization_type=None):
        self.df = df
        self.data_type = data_type
        self.visualization_type = visualization_type
        self.col_names = []
        self.folder_name = 'none'

    def show_graphs(self):
        length_pointer = self.visualization_type_check()
        if length_pointer == 0:
            if self.visualization_type == 'random':
                self.show_graphs_random(sample_size=25)
            else:
                print('Done')
        elif length_pointer > 1:
            self.show_graphs_more_vis_type()
        else:
            self.show_graphs_single_vis_type()

    def error_message(self):
        # Printed if an unknown visualization type is present in the visualization type list
        print(str(self.visualization_type) + ' is not a known visualization type')
        print('The possible options are:')
        for col_name in self.col_names:
            print(col_name)

    def visualization_type_check(self):
        length_pointer = 0
        # col_names is a list with all column names (all possible visualization types)
        self.col_names = self.df.columns.values.tolist()
        # If a correct visualization type is used, get all unique values from this column, to be able to make a graph per
        # value
        if self.visualization_type == 'random':
            self.visualization_type == 'random'
            self.folder_name = self.visualization_type
        elif self.visualization_type == 'race':
            self.folder_name = self.visualization_type
            self.visualization_type = ['year', 'contest_cat', 'boat_cat', 'round_cat', 'round_number']
        else:
            self.visualization_type == [self.visualization_type]
        if len(self.visualization_type) > 1 and type(self.visualization_type) != str:
            if self.visualization_type == 'random':
                print('The visualization type is random')
            else:
                for visualization_type in self.visualization_type:
                    if any(visualization_type in col_name for col_name in self.col_names):
                        length_pointer += 1
                    else:
                        self.error_message()
        if any(self.visualization_type[0] in col_name for col_name in self.col_names) and type(self.visualization_type) != str:
            length_pointer += 1
        # If an incorrect visualization type is used, list all possible visualization types as output so the user can
        # choose a different visualization type
        elif self.visualization_type == 'random':
            print('No need to check if random')
        else:
            self.error_message()
        return length_pointer

    def show_graphs_single_vis_type(self):
        """
        :return: nothing, but saves the unique graphs in a file with corresponding name in the figures folder in a folder
         named after the visualization type (folder should be created on beforehand)
        """
        unique_types = self.df[self.visualization_type].unique()
        # iterate over the unique values and creates a graph per value
        for count, unique_types in enumerate(unique_types):
            # type_df is the dataframe for one unique value of he visualization type
            type_df = self.df[self.df[self.visualization_type].astype(str).str.contains(unique_types)]
            # selection_length is the length of the type_df (number of rows) that belong to a certain visualization type
            # value
            section_length = len(type_df[self.visualization_type].values)
            colors = mpl.cm.rainbow(np.linspace(0, 1, section_length))
            fig, ax = plt.subplots()
            # Create a plot per row so each race will be one line with one color
            for i in range(section_length):
                ax.plot(type_df.columns.values[10:], type_df.iloc[i,10:].values, color=colors[i])
            # Save the combined plot showing all races belonging to one visualization type value
            plt.savefig('../figures/' + str(self.visualization_type) + '/' + str(unique_types) + '_' + str(self.data_type) + '_plot.png')
            time = datetime.now().strftime('%d-%m %H:%M:%S')
            print(str(count))
            print('[%s: saved figure]' % time)

    def show_graphs_more_vis_type(self):
        """
        :return: nothing, but saves the unique graphs in a file with corresponding name in the figures folder in a folder
         named after the visualization type (folder should be created on beforehand)
        """
        # groups = self.df.groupby(self.visualization_type).size().reset_index().rename(columns={0: 'count'})
        groups = self.df.groupby(self.visualization_type)
        print(groups)
        count = 0
        for name, group in groups:
            print(name)
            # The groups keep the index they had in the original dataframe, therefore it needs te be reindexed
            group = group.reset_index(drop=True)
            # Number of teams in the group
            section_length = group.shape[0]
            # Colors used in the plot
            colors = mpl.cm.rainbow(np.linspace(0, 1, section_length))
            fig, ax = plt.subplots()
            # Plot all lines from one group
            for i in range(section_length):
                ax.plot(group.columns.values[10:], group.iloc[i, 10:].values, label=group.loc[i, 'countries'], color=colors[i])
            # Create a legend

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), shadow=True)

            # legend = ax.legend(loc='upper right', shadow=True)
            frame = legend.get_frame()
            frame.set_facecolor('0.90')
            for label in legend.get_texts():
                label.set_fontsize('large')
            for label in legend.get_lines():
                label.set_linewidth(1.5)  # the legend line width


            if self.folder_name == 'race':
                name_array = group.loc[0,['year', 'contest', 'boattype', 'round']].values
                name = ''.join([str(x) for x in name_array])
                # Save the figure in the correct folder
            if self.data_type == 'strokes':
                plt.ylabel('stroke pace (strokes/minute)')
            else:
                plt.ylabel('speed (meters/second)')
            plt.xlabel('distance (meters)')
            if self.data_type:
                plt.savefig('../figures/' + 'separate_boats' + '/' + str(name) + '_' + str(self.data_type) + '_plot.png')
            else:
                plt.savefig('../figures/' + self.folder_name + '/' + str(name) + '_' + str(self.data_type) + '_plot.png')

            count += 1
            time = datetime.now().strftime('%d-%m %H:%M:%S')
            print(str(count))
            print('[%s: saved figure]' % time)

    def show_graphs_random(self, sample_size):
        print('creating random graphs of stroke paces')
        for iteration in range(4):
            index = self.df.index
            rand_indexes = np.random.choice(index, size=sample_size)
            colors = mpl.cm.rainbow(np.linspace(0, 1, sample_size))
            fig, ax = plt.subplots()
            for i, rand_index in enumerate(rand_indexes):
                print(i)
                distances = self.df.columns.values[9:]
                values = self.df.iloc[rand_index, 9:].as_matrix()
                ax.plot(distances, values, color=colors[i])
            plt.ylabel('stroke pace (strokes/minute)')
            plt.xlabel('distance (meters)')
            plt.savefig('../figures/' + self.folder_name + '/' + str(sample_size) + '_' + str(self.data_type) + '_plot' + str(iteration) + '.png')
            fig, ax = plt.subplots()
            for i, rand_index in enumerate(rand_indexes):
                print(i)
                distances = self.df.columns.values[9:]
                values = self.df.iloc[rand_index, 9:].as_matrix()
                gradients = self.df.iloc[rand_index, 9:].diff().as_matrix()
                ax.plot(distances, gradients, color=colors[i])
            plt.ylabel('gradient (strokes/minute)')
            plt.xlabel('distance (meters)')
            plt.savefig('../figures/' + self.folder_name + '/' + str(sample_size) + '_' + str(self.data_type) + '_gradient_plot' + str(iteration) + '.png')
            fig, ax = plt.subplots()
            for i, rand_index in enumerate(rand_indexes):
                print(i)
                distances = [1,2,3,4]
                values = self.df.iloc[rand_index, 9:].as_matrix()
                values = [values[0], values[9], values[19], values[29], values[len(values)-1]]
                gradients = [value - values[index-1] for index, value in enumerate(values) if index > 0]
                ax.scatter(distances, gradients, color=colors[i])
            plt.ylabel('gradient (strokes/minute)')
            plt.xlabel('distance interval (meters)')
            labels = [item.get_text() for item in ax.get_xticklabels()]
            labels[1] = '50-500'
            labels[3] = '500-1000'
            labels[5] = '1000-1500'
            labels[7] = '1500-2000'
            ax.set_xticklabels(labels)
            plt.savefig('../figures/' + self.folder_name + '/' + str(sample_size) + '_' + str(
                self.data_type) + '_gradient_500m_plot' + str(iteration) + '.png')
            print('plots created')
