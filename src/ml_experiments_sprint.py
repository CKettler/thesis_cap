from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import numpy as np
import correlation_calculation as cc

class ml_experiments:
    def __init__(self, df_all, ml_label):
        self.df_all = df_all
        self.ml_label = ml_label

    def perform_experiments(self):
        grouped = self.df_all.groupby(by=['country_cat'])
        lin_scores = []
        rf_scores = []
        rfc_scores = []
        svmc_scores = []
        for name, group in grouped:
            x, y, names = self.x_y_extraction_group(group)
            if y.size > 30:
                cv_value = 5
            elif y.size < 25:
                continue
            else:
                cv_value = 2
            if self.ml_label == 'reg':
                lin_scores.append(self.linear_regression(x, y, cv_value))
                # self.neural_network_regression(x, y)
                rf_scores.append(self.random_forest_regression(x, y, cv_value))
            elif self.ml_label == 'class':
                print(y.ravel().tolist())
                number_classes = len(set(y.ravel().tolist()))
                print(number_classes)
                if number_classes > 1:
                    cc.calculate_cor(x, y, names)
                    rfc_scores.append(self.random_forest_classifier(x, y, cv_value))
                    svmc_scores.append(self.svm_classifier(x, y, cv_value))
        print("[Linear Regression mean score: %s]" % np.mean(lin_scores))
        print("[Random Forest Regressor mean score: %s]" % np.mean(rf_scores))
        print("[Random Forest classifier mean score: %s]" % np.mean(rfc_scores))
        print("[SVM mean score: %s]" % np.mean(svmc_scores))

    def x_y_extraction_group(self, group):
        # get all columns that do not contain values from the race itself
        non_race_columns = ['contest_cat', 'round_cat', 'round_number', 'boat_cat', 'start_lane', 'average_rank_team',
                            'variance_rank_team', 'coxswain', 'races_after_day', 'races_before_day', 'after_is_rep',
                            'before_is_rep', 'country_cat']
        # get the rankings in the race from 0-1500 meters
        race_ranking_columns = ['500m_rank', '1000m_rank', '1500m_rank', '500m_rank_first_bool',
                                '1000m_rank_first_bool', '1500m_rank_first_bool']
        approx_cols = [col for col in group if 'approx' in col and 'rank' in col][:-10]
        race_strokes_columns = [col for col in group if 'stroke' in col and not '2000' in col and not '-' in col]
        race_strokes_columns = race_strokes_columns[:-9]
        # print(race_strokes_columns)
        label_column = ['start_sprint']
        y_and_x = group.loc[:, label_column + non_race_columns + race_ranking_columns + race_strokes_columns]
        names = non_race_columns + race_ranking_columns + approx_cols + race_strokes_columns
        # using only the rows where a sprint is actually happening
        if self.ml_label == 'reg':
            y_and_x = y_and_x[y_and_x.start_sprint != -1]
        y_and_x = y_and_x.dropna()
        y = y_and_x.loc[:, label_column].as_matrix()
        if self.ml_label == 'class':
            y = np.asarray([[1] if y_n > 0 else [0] for y_n in y])
        x = y_and_x.loc[:, non_race_columns + race_ranking_columns + race_strokes_columns].as_matrix()
        return x, y, names

    def x_y_extraction(self):
        # get all columns that do not contain values from the race itself
        non_race_columns = ['contest_cat', 'round_cat', 'round_number', 'boat_cat', 'start_lane', 'average_rank_team',
                            'variance_rank_team', 'coxswain', 'races_after_day', 'races_before_day', 'after_is_rep',
                            'before_is_rep', 'country_cat']
        # get the rankings in the race from 0-1500 meters
        race_ranking_columns = ['500m_rank', '1000m_rank', '1500m_rank', '500m_rank_first_bool',
                                '1000m_rank_first_bool', '1500m_rank_first_bool']
        approx_cols = [col for col in self.df_all.columns if 'approx' in col and 'rank' in col][:-10]
        race_strokes_columns = [col for col in self.df_all.columns if 'stroke' in col and not '2000' in col and not '-' in col]
        race_strokes_columns = race_strokes_columns[:-9]
        # print(race_strokes_columns)
        label_column = ['start_sprint']
        y_and_x = self.df_all.loc[:, label_column + non_race_columns + race_ranking_columns + race_strokes_columns]
        names = non_race_columns + race_ranking_columns + approx_cols + race_strokes_columns
        # using only the rows where a sprint is actually happening
        if self.ml_label == 'reg':
            y_and_x = y_and_x[y_and_x.start_sprint != -1]
        y_and_x = y_and_x.dropna()
        y = y_and_x.loc[:, label_column].as_matrix()
        if self.ml_label == 'class':
            y = np.asarray([[1] if y_n > 0 else [0] for y_n in y])
        x = y_and_x.loc[:, non_race_columns + race_ranking_columns + race_strokes_columns].as_matrix()
        return x, y, names

    # The Regressors

    def linear_regression(self, x, y, cv_value):
        lin_model = linear_model.LinearRegression()
        lin_scores = cross_val_score(lin_model, x, y, cv=cv_value, n_jobs=1)
        # print("[Linear Regression mean score: %s]" % np.mean(lin_scores))
        # print("[Linear Regression standard deviation scores: %s]" % np.std(lin_scores))
        return np.mean(lin_scores)

    def neural_network_regression(self, x, y, cv_value):
        nn_model = MLPRegressor()
        nn_scores = cross_val_score(nn_model, x, y, cv=cv_value, n_jobs=1)
        # print("[Multi Layer Perceptron Regressor mean score: %s]" % np.mean(nn_scores))
        # print("[Multi Layer Perceptron Regressor standard deviation scores: %s]" % np.std(nn_scores))
        return np.mean(nn_scores)

    def random_forest_regression(self, x, y, cv_value):
        rf_model = RandomForestRegressor()
        rf_scores = cross_val_score(rf_model, x, y.ravel(), cv=cv_value, n_jobs=1)
        # print("[Random Forest Regressor mean score: %s]" % np.mean(rf_scores))
        # print("[Random Forest Regressor standard deviation scores: %s]" % np.std(rf_scores))
        return np.mean(rf_scores)

    # The classifiers

    def random_forest_classifier(self, x, y, cv_value):
        rf_class_model = RandomForestClassifier()
        rf_class_scores = cross_val_score(rf_class_model, x, y.ravel(), cv=cv_value, n_jobs=1)
        # print("[Random Forest classifier mean score: %s]" % np.mean(rf_class_scores))
        # print("[Random Forest classifier standard deviation scores: %s]" % np.std(rf_class_scores))
        return np.mean(rf_class_scores)

    def svm_classifier(self, x, y, cv_value):
        svm_model = SVC()
        svm_scores = cross_val_score(svm_model, x, y.ravel(), cv=cv_value, n_jobs=1)
        # print("[SVM mean score: %s]" % np.mean(svm_scores))
        # print("[SVM standard deviation scores: %s]" % np.std(svm_scores))
        return np.mean(svm_scores)
