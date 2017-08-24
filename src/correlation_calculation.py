from scipy.stats import spearmanr
from scipy.stats import pearsonr
import numpy as np
import math

def calculate_cor(x, y, names, y_names):
    no_features = x.shape[1]
    no_classes = y.shape[1]
    correlations = []
    cors = []
    p_values = []
    p_values_inc_names = []
    high_count = 0
    high_values = []
    all_count = 0
    for j in range(no_classes):
        for i in range(no_features):
            # if j == 0:
            all_count += 1
            feature = x[:, i]
            correlation = pearsonr(feature, y[:,j])
            # correlation = spearmanr(feature, y[:, j])
            correlations.append([correlation[0], names[i], y_names[j]])
            cors.append(correlation[0])
            p_values.append(correlation[1])
            p_values_inc_names.append([correlation[1], names[i], y_names[j]])
            # if np.abs(correlation[0]) > 0.2 and correlation[1] < 0.05:
            #     print('high corr: %s' % correlation[0])
            #     print('corresponding p-value: %s' % correlation[1])
            #     high_count += 1
            #     high_values.append(np.abs(correlation[0]))
    # sorted_cors = sorted(correlations, key=lambda x: x[0])
    # print(sorted_cors)
    for index, cor in enumerate(correlations):
        print(cor[1] + " & " + str(cor[0]) + " & " + str(p_values[index]))
    # print(correlations)
    # print(p_values_inc_names)
    cors = [x for x in cors if str(x) != 'nan']
    p_values = [x for x in p_values if str(x) != 'nan']
    # print('mean correlation: %s' % np.mean(cors))
    # print('mean p-value %s' % np.mean(p_values))
    # print('var correlation: %s' % np.var(cors))
    # print('var p-values %s' % np.var(p_values))
    # print('The highest correlation is: %s' % np.max(cors))
    # print('percentage of high correlations: %s' % (float(high_count) / float(all_count)))
    # print('high values average: %s' % np.mean(high_values))

def calculate_same_cor(x, y, names, y_names):
    no_features = x.shape[1]
    no_classes = y.shape[1]
    correlations = []
    for i in range(no_features):
        feature = x[:, i]
        correlation = spearmanr(feature, y[:, i])
        correlations.append([correlation[0], names[i], y_names[i]])
    sorted_cors = sorted(correlations, key=lambda x: x[0])
    print(sorted_cors)
    return sorted_cors

def calculate_line_cor(x,y):
    no_features = x.shape[1]
    no_instances = x.shape[0]
    correlations = []
    p_values = []
    high_count = 0
    all_count = 0
    high_values = []
    for i in range(no_instances):
        all_count += 1
        feature = x[i, :]
        y_values = y[i, :]
        if np.isnan(np.min(y_values)):
            continue
        print(i)
        correlation = spearmanr(feature, y_values)
        if np.var(y[i, :]) == 0:
            print('y: %s' % y[i, :])
            continue
        # if math.isnan(correlation[0]):
        #     print('x: %s' % feature)
        #     print('y: %s' % y[i, :])
        if np.abs(correlation[0]) > 0.2 and correlation[1] < 0.05:
            print('high corr: %s' % correlation[0])
            print('corresponding p-value: %s' % correlation[1])
            high_count += 1
            high_values.append(np.abs(correlation[0]))
        correlations.append(correlation[0])
        p_values.append(correlation[1])
    correlations = [x for x in correlations if str(x) != 'nan']
    p_values = [x for x in p_values if str(x) != 'nan']
    mean_correlation = np.mean(correlations)
    mean_p_value = np.mean(p_values)
    print('The mean correlation is: %s' % mean_correlation)
    print('The mean p-value is: %s' % mean_p_value)
    print('The min p-value is: %s' % np.min(p_values))
    print('The highest correlation is: %s' % np.max(correlations))
    print('var correlation: %s' % np.var(correlations))
    print('var p-values %s' % np.var(p_values))
    print('percentage of high correlations: %s' % (float(high_count)/float(all_count)))
    print('high values average: %s' % np.mean(high_values))


def calculate_line_avg_cor(x,y):
    no_features = x.shape[1]
    no_instances = x.shape[0]
    correlations = []
    p_values = []
    high_count = 0
    all_count = 0
    high_values = []
    for i in range(no_instances):
        all_count += 1
        feature = []
        y_values = []
        part_x = []
        part_y = []
        for t in range(40):
            part_x.append(x[i, t])
            part_y.append(y[i, t])
            if not t%10:
                feature.append(np.mean(part_x))
                part_x = []
                y_values.append(np.mean(part_y))
                part_y = []
        if np.isnan(np.min(y_values)):
            continue
        correlation = spearmanr(feature, y_values)
        if np.var(y[i, :]) == 0:
            print('y: %s' % y[i, :])
            continue
        # if math.isnan(correlation[0]):
        #     print('x: %s' % feature)
        #     print('y: %s' % y[i, :])
        if np.abs(correlation[0]) > 0.2 and correlation[1] < 0.05:
            print('high corr: %s' % correlation[0])
            # print('corresponding p-value: %s' % correlation[1])
            high_count += 1
            high_values.append(correlation[0])
        correlations.append(correlation[0])
        p_values.append(correlation[1])
    correlations = [x for x in correlations if str(x) != 'nan']
    p_values = [x for x in p_values if str(x) != 'nan']
    mean_correlation = np.mean(correlations)
    mean_p_value = np.mean(p_values)
    print('The mean correlation is: %s' % mean_correlation)
    print('The mean p-value is: %s' % mean_p_value)
    print('The min p-value is: %s' % np.min(p_values))
    print('The highest correlation is: %s' % np.max(correlations))
    print('var correlation: %s' % np.var(correlations))
    print('var p-values %s' % np.var(p_values))
    print('percentage of high correlations: %s' % (float(high_count)/float(all_count)))
    print('high values average: %s' % np.mean(high_values))
