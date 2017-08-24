import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

class strat_plotting:
    def __init__(self, strategies, strokes, max_strokes, min_strokes, name):
        # self.values_per_box = values_per_box
        # self.most_used_strategy = most_used_strat
        # self.plot_indicator = plot_indicator
        self.strategies = strategies
        self.strokes = strokes
        self.max_strokes = max_strokes
        self.min_strokes = min_strokes
        self.name = name

    def plot_strategies(self):
        nr_strategies = len(self.strategies)
        # colors = mpl.cm.rainbow(np.linspace(0, 1, nr_strategies))
        distances = np.array([500 * x if x > 0 else 50 for x in range(0, 5)])
        fig_name = self.name + '.png'
        high_line, low_line = self.make_high_low_line(self.max_strokes, self.min_strokes)
        self.plot_strategy_area(distances, self.strokes[0], high_line, low_line, fig_name)
        # fig, ax = plt.subplots()
        # for i, strokes in enumerate(self.strokes):
        #     # print(i)
        #     # strategy = self.gradient_to_strat(strategy)
        #     ax.plot(distances, strokes, color=colors[i])
        # plt.ylabel('normalized strokes/minute')
        # plt.xlabel('distance (meters)')
        # plt.savefig(fig_name)

    def make_high_low_line(self, max_strokes, min_strokes):
        high_line = []
        low_line = []
        for i in range(5):
            if max_strokes[i] >= min_strokes[i]:
                high_line.append(max_strokes[i])
                low_line.append(min_strokes[i])
            else:
                high_line.append(min_strokes[i])
                low_line.append(max_strokes[i])
        return high_line, low_line

    def plot_strategy_area(self, x, y_average, y_max, y_min, fig_name):
        fig, ax = plt.subplots()
        ax.plot(x, y_average, color='black')
        ax.plot(x, y_max, color='blue', alpha=0.5)
        ax.plot(x, y_min, color='blue', alpha=0.5)
        ax.fill_between(x, y_average, y_max, facecolor='blue', alpha=0.5)
        ax.fill_between(x, y_average, y_min, facecolor='blue', alpha=0.5)
        ax.set_ylabel('normalized strokes/minute')
        ax.set_xlabel('distance (meters)')
        plt.savefig(fig_name)

    def gradient_to_strat(self, strategy):
        strat_list = [1]
        for i in range(0,4):
            if i == 0:
                strat_list.append(1+strategy[i])
            else:
                strat_list.append(strat_list[i]+strategy[i])
        return strat_list



            # def plot_strategies(self):
    #     # print('strategy %s' % self.most_used_strategy)
    #     parts = ['1', '2', '3', '4']
    #     strategy_boxplot = False
    #     lower_bound_diff, upper_bound_diff, box_upper_line, box_lower_line, values = \
    #         self.strategy_to_pace_diff(self.most_used_strategy, strategy_boxplot)
    #     if self.plot_indicator:
    #         self.box_plot(values, parts, strategy_boxplot)
    #     strategy_boxplot = True
    #     lower_bound_diff, upper_bound_diff, box_upper_line, box_lower_line, values = \
    #         self.strategy_to_pace_diff(self.most_used_strategy, strategy_boxplot)
    #     if self.plot_indicator:
    #         self.box_plot(values, parts, strategy_boxplot)
    #     start_pace = 45
    #     distace = [0, 500, 1000, 1500, 2000]
    #     lower_bound_line = self.diff_to_pace(start_pace, lower_bound_diff)
    #     upper_bound_line = self.diff_to_pace(start_pace, upper_bound_diff)
    #     box_lower_line = self.diff_to_pace(start_pace, box_lower_line)
    #     box_upper_line = self.diff_to_pace(start_pace, box_upper_line)
    #     if self.plot_indicator:
    #         self.plot_strategy_area(distace, lower_bound_line, upper_bound_line, box_lower_line, box_upper_line)
    #     else:
    #         return box_lower_line, box_upper_line
    #
    # def box_plot(self, values, parts, strategy_boxplot):
    #     fig, ax = plt.subplots()
    #     # rectangular box plot
    #     bplot = ax.boxplot(values,
    #                              vert=True,  # vertical box alignment
    #                              patch_artist=True,  # fill with color
    #                              labels=parts)  # will be used to label x-ticks
    #     if strategy_boxplot:
    #         ax.set_title('Most used race strategy box plot')
    #     else:
    #         ax.set_title('Strategy box plot')
    #
    #     # fill with colors
    #     for patch in bplot['boxes']:
    #         patch.set_facecolor('lightgreen')
    #
    #     # adding horizontal grid lines
    #     ax.yaxis.grid(True)
    #     ax.set_xlabel('Four race parts')
    #     ax.set_ylabel('Observed values of gradient')
    #     if strategy_boxplot:
    #         plt.savefig("../figures/MUS_boxplot.png")
    #     else:
    #         plt.savefig("../figures/all_strat_boxplot.png")

    # def strategy_to_pace_diff(self, strategy, strategy_boxplot):
    #     upper_bound_line = []
    #     lower_bound_line = []
    #     box_upper_line = []
    #     box_lower_line = []
    #     values = []
    #     if strategy_boxplot:
    #         for index, part in enumerate(strategy):
    #             race_part_boxes = self.values_per_box[index]
    #             if part == -3:
    #                 values.append(race_part_boxes[0])
    #                 values_array = np.array(race_part_boxes[0])
    #                 upper_bound_line.append(np.amax(values_array))
    #                 lower_bound_line.append(np.amin(values_array))
    #                 box_upper_line.append(np.percentile(values_array, 75))
    #                 box_lower_line.append(np.percentile(values_array, 25))
    #             elif part == -2:
    #                 values.append(race_part_boxes[1])
    #                 values_array = np.array(race_part_boxes[1])
    #                 upper_bound_line.append(np.amax(values_array))
    #                 lower_bound_line.append(np.amin(values_array))
    #                 box_upper_line.append(np.percentile(values_array, 75))
    #                 box_lower_line.append(np.percentile(values_array, 25))
    #             elif part == -1:
    #                 values.append(race_part_boxes[2])
    #                 values_array = np.array(race_part_boxes[2])
    #                 upper_bound_line.append(np.amax(values_array))
    #                 lower_bound_line.append(np.amin(values_array))
    #                 box_upper_line.append(np.percentile(values_array, 75))
    #                 box_lower_line.append(np.percentile(values_array, 25))
    #             elif part == 0:
    #                 values.append(race_part_boxes[3])
    #                 values_array = np.array(race_part_boxes[3])
    #                 upper_bound_line.append(np.amax(values_array))
    #                 lower_bound_line.append(np.amin(values_array))
    #                 box_upper_line.append(np.percentile(values_array, 75))
    #                 box_lower_line.append(np.percentile(values_array, 25))
    #             elif part == 1:
    #                 values.append(race_part_boxes[4])
    #                 values_array = np.array(race_part_boxes[4])
    #                 upper_bound_line.append(np.amax(values_array))
    #                 lower_bound_line.append(np.amin(values_array))
    #                 box_upper_line.append(np.percentile(values_array, 75))
    #                 box_lower_line.append(np.percentile(values_array, 25))
    #             elif part == 2:
    #                 values.append(race_part_boxes[5])
    #                 values_array = np.array(race_part_boxes[5])
    #                 upper_bound_line.append(np.amax(values_array))
    #                 lower_bound_line.append(np.amin(values_array))
    #                 box_upper_line.append(np.percentile(values_array, 75))
    #                 box_lower_line.append(np.percentile(values_array, 25))
    #             elif part == 3:
    #                 values.append(race_part_boxes[6])
    #                 values_array = np.array(race_part_boxes[6])
    #                 upper_bound_line.append(np.amax(values_array))
    #                 lower_bound_line.append(np.amin(values_array))
    #                 box_upper_line.append(np.percentile(values_array, 75))
    #                 box_lower_line.append(np.percentile(values_array, 25))
    #     else:
    #         for index in range(4):
    #             race_part_boxes = self.values_per_box[index]
    #             values_per_part = []
    #             for box in range(7):
    #                 values_per_part += race_part_boxes[box]
    #             values.append(values_per_part)
    #     return lower_bound_line, upper_bound_line, box_upper_line, box_lower_line, values

    # def diff_to_pace(self, start_pace, line_diff):
    #     pace = [start_pace]
    #     for i, diff in enumerate(line_diff):
    #         if i == 0:
    #             new_pace = start_pace + diff
    #         else:
    #             new_pace += diff
    #         pace.append(new_pace)
    #     return pace





