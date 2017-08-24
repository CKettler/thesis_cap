import pandas as pd
import correlation_calculation as cc
import statistical_tests as st
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
import minirank as mr
import scipy.stats as stats
import pylab as pl
import sys
from datetime import datetime


class ml_experiments_strat:
    def __init__(self, df_all):
        self.df_all = df_all
        self.same_test = False
        self.line_cor = False
        self.averaging = False
        self.strategy_determination = False
        self.group_prediction = False

    def experiments(self):
        self.group_prediction = False
        time = str(datetime.now().time())
        time = time.replace(":", "")
        time = time.replace(".", "")
        # orig_stdout = sys.stdout
        # f = open(str(time) + '_out.txt', 'w')
        # sys.stdout = f
        if self.strategy_determination:
            n_loops = 4
        else:
            n_loops = 1
        for y_index in range(0,n_loops):
            x, y, x_names, y_names = self.labeling(y_index)
            if self.strategy_determination:
                x, x_names = self.x_to_one_hot(x, x_names)
            if self.same_test:
                same_cor = cc.calculate_same_cor(x, y, x_names, y_names)
                same_permutation_list = []
                for i in range(x.shape[1]):
                    statistical_test = st.statistical_tests(x[:, i], y[:, i], nmc=30000)
                    same_permutation_list.append(statistical_test.exact_mc_perm_test())
                print(same_permutation_list)
                for i in range(x.shape[1]):
                    pl.figure()
                    pl.hist(x[:, i], normed=True)  # use this to draw histogram of your data
                    pl.savefig('../figures/feature_distributions/distribution_x_' + x_names[i] + '.png')
                    pl.figure()
                    pl.hist(y[:, i], normed=True)  # use this to draw histogram of your data
                    pl.savefig('../figures/feature_distributions/distribution_y_' + y_names[i] + '.png')
            elif self.line_cor:
                if self.averaging:
                    cc.calculate_line_avg_cor(x, y)
                else:
                    cc.calculate_line_cor(x, y)
            else:
                # y_classes, y_1 = self.divide_classes(y)
                y_classes = y
                # cc.calculate_cor(x, y_classes, x_names, y_names)
                if self.strategy_determination:
                    rf_scores = []
                    ml_count = 0
                    for fold in range(0,1):
                        rf = self.strategy_regression(x, y, x_names, ml_count, y_index)
                        # if fold%10 == 0 and fold != 0:
                        #     ml_count += 1
                        #     # rf_scores = []
                        #     rf = self.strategy_regression(x, y, x_names, ml_count)
                        # rf_scores.append(self.strategy_regression(x, y, x_names, ml_count))
                        # if fold % 10 == 9:
                        #     print('mean rf score: %s' % np.mean(rf_scores))
                else:
                    self.ordinal_rank_classification(x, y, 10)
                    # self.knn_rank_classification(x, y)
                    # y_classes, y_1 = self.divide_classes(y)
                    # cc.calculate_cor(x, y, x_names, y_names)
                    # self.ordinal_rank_classification(x, y, 10)
                    # self.rank_classification(x, y_classes)
                    # self.first_classification(x, y_1, x_names)

        # sys.stdout = orig_stdout
        # f.close()

    def experiments_grouping(self):
        self.group_prediction = True
        x_names = ['50-450_slope_cat', '450-950_slope_cat', '950-1450_slope_cat', '1450-1950_slope_cat']
        y_name = ['2000m_rank']
        group_cols = ['boatsize', 'races_after_day', 'round_cat']
        considered_configs = ['1H0', '1S0', '2S0', '4S0', '4F0']
        prediction_groups = self.df_all.groupby(group_cols)
        for name, group in prediction_groups:
            boatsize = group.loc[group.index[0],'boatsize']
            group_round = group.loc[group.index[0],'round'][0]
            races_after = group.loc[group.index[0],'races_after_day']
            config = str(boatsize) + str(group_round) + str(races_after)
            # if config not in considered_configs:
            #     continue
            # if config != '1H0':
            #     continue
            print('Current Configuration: %s' % config)
            print('Used sample size: %s' % group.shape[0])
            x_y = group.loc[:, x_names + y_name]
            x_y = x_y.dropna()
            x = x_y.loc[:, x_names]
            y = x_y.loc[:, y_name]
            x = x.as_matrix()
            y = y.as_matrix()
            # self.ordinal_rank_classification(x, y, 50)
            cc.calculate_cor(x, y, x_names, y_name)
            # self.rank_classification(x, y)
            # self.knn_rank_classification(x, y)

    def labeling(self, y_index):
        # x_names = ['round_cat', 'boatsize', 'team_cat', 'races_after_day']
        if self.strategy_determination:
            x_names = ['round_cat', 'boatsize', 'team_cat', 'races_after_day']
            if y_index == 0:
                y_name = ['50-450_slope_cat']
            elif y_index == 1:
                y_name = ['450-950_slope_cat']
            elif y_index == 2:
                y_name = ['950-1450_slope_cat']
            elif y_index == 3:
                y_name = ['1450-1950_slope_cat']
        else:
            x_names = ['50-450_slope_cat', '450-950_slope_cat', '950-1450_slope_cat', '1450-1950_slope_cat']
            y_name = ['2000m_rank']
        x_y = self.df_all.loc[:, x_names + y_name]
        x_y = x_y.dropna()
        # if 'average_rank_opponents' in x_names:
        #     nan_truth = np.isnan(x.loc[:, 'average_rank_opponents'].as_matrix())
        #     for i, truth in enumerate(nan_truth):
        #         if truth:
        #             print('index of nan is: %s' %i)
        #             # x.loc[i, 'average_rank_opponents'] = 0
        #             x = x.drop([i])
        #             y = y.drop([i])
        x = x_y.loc[:, x_names]
        y = x_y.loc[:, y_name]
        x = x.as_matrix()
        y = y.as_matrix()
        return x, y, x_names, y_name

    def x_to_one_hot(self, all_x, all_x_names):
        for i, x in enumerate(np.transpose(all_x)):
            if len(all_x_names) > 1:
                x_name = all_x_names[i]
            else:
                x_name = all_x_names
            x, x_names = self.make_one_hot(x, x_name)
            if i == 0:
                all_x_new = x
                all_x_names_new = x_names
            else:
                all_x_new = np.hstack([all_x_new, x])
                all_x_names_new = np.hstack([all_x_names_new, x_names])
        return all_x_new, all_x_names_new

    def make_one_hot(self, x, x_name):
        print('highest class is: %s' % np.amax(x))
        print('lowest class is: %s' % np.amin(x))
        if np.amax(x) == 1 and np.amin(x) == 0:
            return np.array([x]).transpose(), x_name
        all_x = []
        new_x_names = []
        for class_name in range(int(np.amin(x)), int(np.amax(x)) + 1):
            if class_name in x:
                new_x = []
                new_name = x_name + '_' + str(class_name)
                for x_value in x:
                    if x_value == class_name:
                        new_x.append(1)
                    else:
                        new_x.append(0)
                all_x.append(new_x)
                new_x_names.append(new_name)
        return np.array(all_x).transpose(), new_x_names

    def divide_classes(self, y):
        print('highest class is: %s' % np.amax(y))
        print('lowest class is: %s' % np.amin(y))
        all_y = []
        for rank in range(int(np.amin(y)), int(np.amax(y))+1):
            new_y = []
            for y_value in y:
                if y_value == rank:
                    new_y.append(1)
                else:
                    new_y.append(0)
            all_y.append(new_y)
        first_y = all_y[0]
        return np.array(all_y).transpose(), np.array([first_y]).transpose()

    def knn_rank_classification(self, x, y, n_folds=10):
        score_knn = []
        nr_features = x.shape[1]
        y = np.reshape(y, (y.shape[0], 1))
        all = np.hstack([x, y])
        np.random.shuffle(all)
        x_shuf = all[:, :nr_features]
        y_shuf = all[:, nr_features:]
        train_data_x = x_shuf[:3500, :]
        train_data_y = np.ravel(y_shuf[:3500, :])
        test_data_x = x_shuf[3500:, :]
        test_data_y = np.ravel(y_shuf[3500:, :])
        print('K Nearest Neighbors')
        parameter_values = [x for x in range(300) if x%100 != 0]
        parameters = {'n_estimators': parameter_values}
        forest = RandomForestRegressor()
        clf = GridSearchCV(forest, parameters, cv=10)
        clf.fit(train_data_x, train_data_y)
        mean_scores = np.array(clf.cv_results_['mean_test_score']).tolist()
        print(parameter_values)
        print(mean_scores)
        # for n in range(n_folds):
        #     nr_features = x.shape[1]
        #     nr_instances = x.shape[0]
        #     if self.group_prediction:
        #         threshold = int((2 * nr_instances) / 3)
        #     else:
        #         threshold = 3500
        #     y = np.reshape(y, (y.shape[0], 1))
        #     all = np.hstack([x, y])
        #     np.random.shuffle(all)
        #     x_shuf = all[:, :nr_features]
        #     y_shuf = all[:, nr_features:]
        #     train_data_x = x_shuf[:threshold, :]
        #     train_data_y = np.ravel(y_shuf[:threshold, :])
        #     test_data_x = x_shuf[threshold:, :]
        #     test_data_y = np.ravel(y_shuf[threshold:, :])
        #     print('K Nearest Neighbors')
        #     clf = KNeighborsRegressor(n_neighbors=20)
        #     model = clf.fit(train_data_x, train_data_y)
        #     pred = model.predict(test_data_x)
        #     s = metrics.mean_absolute_error(test_data_y, pred)
        #     r2 = metrics.r2_score(test_data_y, pred)
        #     print('R^2 (K-NN)  fold %s: %s' % (n + 1, r2))
        #     score_knn.append(r2)
            # mean_scores = np.array(clf.cv_results_['mean_test_score'])
            # print(parameter_values)
            # print(mean_scores)
        # print('MEAN ABSOLUTE ERROR (K-NN):    %s' % np.mean(score_knn))
        # print('R2 (K-NN):    %s' % np.mean(score_knn))

    def rank_classification(self, x, y):
        n_classes = y.shape[1]
        rf_scores = []
        mae_scores = []
        for n in range(1):
            nr_features = x.shape[1]
            nr_instances = x.shape[0]
            y = np.reshape(y, (y.shape[0], 1))
            all = np.hstack([x, y])
            np.random.shuffle(all)
            x_shuf = all[:, :nr_features]
            y_shuf = all[:, nr_features:]
            if self.group_prediction:
                threshold = int((2 * nr_instances) / 3)
            else:
                threshold = 3500
            print('training data size: %s ' % threshold)
            print('test data size: %s ' % (nr_instances - threshold))
            train_data_x = x_shuf[:threshold, :]
            train_data_y = np.ravel(y_shuf[:threshold, :])
            test_data_x = x_shuf[threshold:, :]
            test_data_y = np.ravel(y_shuf[threshold:, :])
            # forest = RandomForestClassifier(n_estimators=100, random_state=1)
            # multi_target_forest = MultiOutputClassifier(forest, n_jobs=-1)
            # model = multi_target_forest.fit(train_data_x, train_data_y)
            # parameter_values = ['identity', 'logistic', 'tanh', 'relu']
            # parameters = {'activation': parameter_values}
            # forest = MLPRegressor(solver='lbfgs')
            # clf = GridSearchCV(forest, parameters, cv=10)
            # clf.fit(x, y)
            # mean_scores = np.array(clf.cv_results_['mean_test_score'])
            # print('ACTIVATION')
            # print(parameter_values)
            # print(mean_scores)
            # parameter_values = ['lbfgs', 'adam', 'sgd']
            # parameters = {'solver': parameter_values}
            # forest = MLPRegressor()
            # clf = GridSearchCV(forest, parameters, cv=10)
            # clf.fit(x, y)
            # mean_scores = np.array(clf.cv_results_['mean_test_score'])
            # print('OPTIMIZER')
            # print(parameter_values)
            # print(mean_scores)
            # parameter_values = [(25,),(25,25),(25,25,25), (50,), (50,50), (50,50,50), (75,), (75,75), (75,75,75),
            #                     (100,), (100,100), (100,100,100), (125,), (125,125), (125,125,125), (150,), (150,150),
            #                     (150,150,150)]
            # parameters = {'hidden_layer_sizes': parameter_values}
            # forest = MLPRegressor(solver='adam', activation='relu')
            # clf = GridSearchCV(forest, parameters, cv=10)
            # clf.fit(x, y)
            # mean_scores = np.array(clf.cv_results_['mean_test_score'])
            # print(parameter_values)
            # print(mean_scores)
            forest = MLPRegressor(activation='relu', solver='adam')
            model = forest.fit(train_data_x, train_data_y)
            pred = forest.predict(test_data_x)
            if n == 0:
                comparing_histogram_plot(pred, test_data_y, method='MLP')
            mae = metrics.mean_absolute_error(test_data_y, pred)
            rf_scores.append(metrics.r2_score(test_data_y, pred))
            mae_scores.append(mae)
        print('r2 MLP: %s' % np.mean(rf_scores))
        print('MAE MLP: %s' % np.mean(mae_scores))

    def ordinal_rank_classification(self, x, y, nfolds):
        score_ordinal_logistic = []
        r2_ordinal_logistic = []
        score_logistic = []
        r2_logistic = []
        score_ridge = []
        r2_ridge = []
        score_knn = []
        r2_knn = []
        for n in range(nfolds):
            nr_features = x.shape[1]
            nr_instances = x.shape[0]
            y = np.reshape(y, (y.shape[0], 1))
            all = np.hstack([x, y])
            np.random.shuffle(all)
            x_shuf = all[:, :nr_features]
            y_shuf = all[:, nr_features:]
            if self.group_prediction:
                threshold = int((2*nr_instances)/3)
            else:
                threshold = 3500
            # print('training data size: %s ' % threshold)
            # print('test data size: %s ' % (nr_instances - threshold))
            train_data_x = x_shuf[:threshold, :]
            train_data_y = np.ravel(y_shuf[:threshold, :])
            test_data_x = x_shuf[threshold:, :]
            test_data_y = np.ravel(y_shuf[threshold:, :])
            # print('train_data_x shape: %s' % train_data_x.shape[0])
            # print('train_data_y shape: %s' % train_data_y.shape[0])
            w, theta = mr.ordinal_logistic_fit(train_data_x, train_data_y)
            pred = mr.ordinal_logistic_predict(w, theta, test_data_x)
            if n == 0:
                comparing_histogram_plot(pred, test_data_y, method='Ordinal Logistic Regression')
            s = metrics.mean_absolute_error(test_data_y, pred)
            r_2 = metrics.r2_score(test_data_y, pred)
            # print('ERROR (ORDINAL)  fold %s: %s' % (n + 1, s))
            # print('ERROR (ORDINAL)  fold %s: %s' % (n + 1, r_2))
            score_ordinal_logistic.append(s)
            r2_ordinal_logistic.append(r_2)
            clf = RandomForestRegressor(n_estimators=100)
            clf.fit(train_data_x, train_data_y)
            pred = clf.predict(test_data_x)
            if n == 0:
                comparing_histogram_plot(pred, test_data_y, method='Random Forest Regression')
            s = metrics.mean_absolute_error(test_data_y, pred)
            r_2 = metrics.r2_score(test_data_y, pred)
            # print('ERROR (LOGISTIC) fold %s: %s' % (n+1, s))
            # print('ERROR (LOGISTIC)  fold %s: %s' % (n + 1, r_2))
            score_logistic.append(s)
            r2_logistic.append(r_2)
            clf = MLPRegressor(solver='lbfgs', activation='logistic', hidden_layer_sizes=(150,150))
            clf.fit(train_data_x, train_data_y)
            pred = np.round(clf.predict(test_data_x))
            if n == 0:
                comparing_histogram_plot(pred, test_data_y, method='MLP')
            s = metrics.mean_absolute_error(test_data_y, pred)
            r_2 = metrics.r2_score(test_data_y, pred)
            # print('ERROR (RIDGE)    fold %s: %s' % (n+1, s))
            # print('R2 (RIDGE)  fold %s: %s' % (n + 1, r_2))
            score_ridge.append(s)
            r2_ridge.append(r_2)
            # parameter_values = [3,5,7,9,11,13,15,17]
            # parameters = {'n_neighbors': parameter_values}
            clf = KNeighborsRegressor(n_neighbors=20)
            # clf = GridSearchCV(forest, parameters, cv=10)
            model = clf.fit(train_data_x, train_data_y)
            pred = model.predict(test_data_x)
            if n == 0:
                comparing_histogram_plot(pred, test_data_y, method='K-Nearest Neighbors')
            s = metrics.mean_absolute_error(test_data_y, pred)
            r_2 = metrics.r2_score(test_data_y, pred)
            # print('ERROR (K-NN)  fold %s: %s' % (n + 1, s))
            # print('R2 (K-NN)  fold %s: %s' % (n + 1, r_2))
            score_knn.append(s)
            r2_knn.append(r_2)
            # mean_scores = np.array(clf.cv_results_['mean_test_score'])
            # print(parameter_values)
            # print(mean_scores)
        # print('MEAN ABSOLUTE ERROR (K-NN):    %s' % np.mean(score_knn))
        print('R2 (KNN):     %s' % np.mean(r2_knn))
        # print('MEAN ABSOLUTE ERROR (ORDINAL LOGISTIC):    %s' % np.mean(score_ordinal_logistic))
        print('R2 (ORDINAL LOGISTIC):     %s' % np.mean(r2_ordinal_logistic))
        # print('MEAN ABSOLUTE ERROR (LOGISTIC REGRESSION): %s' % np.mean(score_logistic))
        print('R2 (RANDOM FOREST REGRESSION):     %s' % np.mean(r2_logistic))
        # print('MEAN ABSOLUTE ERROR (RIDGE REGRESSION):    %s' % np.mean(score_ridge))
        print('R2 (MLP REGRESSION):     %s' % np.mean(r2_ridge))
        # print('Chance level is at %s' % (1. / np.unique(y).size))

    def first_classification(self, x, y, x_names):
        nr_features = x.shape[1]
        print(x.shape)
        y = np.reshape(y, (y.shape[0],1))
        print(y.shape)
        all = np.hstack([x, y])
        np.random.shuffle(all)
        x_shuf = all[:,:nr_features]
        y_shuf = all[:,nr_features:]
        train_data_x = x_shuf[:3500,:]
        train_data_y = np.ravel(y_shuf[:3500,:])
        test_data_x = x_shuf[3500:,:]
        test_data_y = np.ravel(y_shuf[3500:,:])
        forest = RandomForestClassifier(n_estimators=200, random_state=1)
        model = forest.fit(train_data_x, train_data_y)
        rf_scores = model.score(test_data_x, test_data_y)
        feature_importances = model.feature_importances_
        feature_importances = [(feature, x_names[i]) for i, feature in enumerate(feature_importances)]
        sorted_imps = sorted(feature_importances, key=lambda x: x[0])
        # rf_scores = cross_val_score(forest, x, y, cv=5, n_jobs=1)
        print(sorted_imps)
        print('only first prediction rf_scores : %s' % rf_scores)


    def strategy_regression(self, x, y, x_names, ml_count, y_index):
        nr_features = x.shape[1]
        y = np.reshape(y, (y.shape[0], 1))
        all = np.hstack([x, y])
        np.random.shuffle(all)
        x_shuf = all[:, :nr_features]
        y_shuf = all[:, nr_features:]
        train_data_x = x_shuf[:3500, :]
        train_data_y = np.ravel(y_shuf[:3500, :])
        test_data_x = x_shuf[3500:, :]
        test_data_y = np.ravel(y_shuf[3500:, :])
        if y_index == 0:
            n_trees = 40
            hidden_layer_size = (125,)
            optimizer = 'adam'
        elif y_index == 1:
            n_trees = 40
            hidden_layer_size = (50,)
            optimizer = 'lbfgs'
        elif y_index == 2:
            n_trees = 30
            hidden_layer_size = (150,150)
            optimizer = 'lbfgs'
        elif y_index == 3:
            n_trees = 40
            hidden_layer_size = (50,)
            optimizer = 'adam'
        else:
            print('non existing race part: %s' % y_index)
            n_trees = 40
            hidden_layer_size = (50,)
        # print('Random Forest Regression')
        # [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        # parameter_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        # parameters = {'n_estimators': parameter_values}
        rf_scores = []
        knn_scores = []
        mlp_scores = []
        for n in range(1):
            forest = RandomForestRegressor(n_estimators=n_trees)
            model = forest.fit(train_data_x, train_data_y)
            rf_scores.append(model.score(test_data_x, test_data_y))
            feature_importances = model.feature_importances_
            feature_importances = [(feature, x_names[i]) for i, feature in enumerate(feature_importances)]
            sorted_imps = sorted(feature_importances, key=lambda x: x[0])
            print(sorted_imps)
            forest = KNeighborsRegressor(n_neighbors=10)
            model = forest.fit(train_data_x, train_data_y)
            knn_scores.append(model.score(test_data_x, test_data_y))
            forest = MLPRegressor(hidden_layer_sizes=hidden_layer_size, activation='logistic', solver=optimizer)
            model = forest.fit(train_data_x, train_data_y)
            mlp_scores.append(model.score(test_data_x, test_data_y))
        print('Random Forest r2 score: %s' % np.mean(rf_scores))
        print('K-Nearest Neighbors r2 score: %s' % np.mean(knn_scores))
        print('MLP r2 score: %s' % np.mean(mlp_scores))
        # clf = GridSearchCV(forest, parameters, cv=10)
        # clf.fit(train_data_x, train_data_y)
        # mean_scores = np.array(clf.cv_results_['mean_test_score'])
        # print(parameter_values)
        # print(mean_scores)
        # print('K Nearest Neighbors')
        # parameter_values = [x for x in range(60) if x%2 != 0]
        # parameters = {'n_neighbors': parameter_values}
        # forest = KNeighborsRegressor()
        # clf = GridSearchCV(forest, parameters, cv=10)
        # clf.fit(train_data_x, train_data_y)
        # mean_scores = np.array(clf.cv_results_['mean_test_score']).tolist()
        # print(parameter_values)
        # print(mean_scores)
        # print('Multi-Layer Perceptron hidden layer')
        # parameter_values = ['lbfgs', 'adam', 'sgd']
        # parameters = {'solver': parameter_values}
        # forest = MLPRegressor()
        # clf = GridSearchCV(forest, parameters, cv=10)
        # clf.fit(train_data_x, train_data_y)
        # mean_scores = np.array(clf.cv_results_['mean_test_score'])
        # print(parameter_values)
        # print(mean_scores)
        # print('Multi-Layer Perceptron hidden layer')
        # parameter_values = [(150,),(150,150),(160,), (160,160), (170,), (170,170), (200,200)]
        # parameters = {'hidden_layer_sizes': parameter_values }
        # forest = MLPRegressor(activation='logistic')
        # clf = GridSearchCV(forest, parameters, cv=10)
        # clf.fit(train_data_x, train_data_y)
        # mean_scores = np.array(clf.cv_results_['mean_test_score'])
        # print(parameter_values)
        # print(mean_scores)
        #     print('AdaBoost Regression')
        #     forest = AdaBoostRegressor(n_estimators=100)
        # elif ml_count == 2:
        #     print('K-Nearest Neighbors')
        #     forest = KNeighborsRegressor(n_neighbors=5)
        # else:
        #     print('Multi-Layer Perceptron')
        #     forest = MLPRegressor(solver=lbfgs)
        # model = forest.fit(train_data_x, train_data_y)
        # rf_scores = model.score(test_data_x, test_data_y)
        # feature_importances = model.feature_importances_
        # feature_importances = [(feature, x_names[i]) for i, feature in enumerate(feature_importances)]
        # sorted_imps = sorted(feature_importances, key=lambda x: x[0])
        # rf_scores = cross_val_score(forest, x, y, cv=1, n_jobs=1)
        # print(sorted_imps)
        rf_scores = []
        return rf_scores


def comparing_histogram_plot(list_group1, list_group2, method):
    group1_name = 'f(x)'
    group2_name = 'y(x)'
    pl.figure()
    bins = np.linspace(-1, 7, 8)
    pl.hist(list_group1, bins=bins, alpha=0.6, label=str(group1_name))
    pl.hist(list_group2, bins=bins, alpha=0.6, label=str(group2_name))
    pl.legend(loc='upper right')
    pl.xlabel('rank')
    pl.ylabel('frequency')
    pl.savefig('../figures/prediction_results/hist_' + method + '.png')
