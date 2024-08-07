from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from pytorch_tabnet.tab_model import TabNetClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from gain import gain
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score


def imputer(data_name):
    gain_parameters = {'batch_size': 128,
                       'hint_rate': 0.9,
                       'alpha': 1e4,
                       'iterations': 10000}

    # Load data
    miss_data_x = np.loadtxt('data/' + data_name + '.csv', delimiter=",", skiprows=1)

    # Impute missing data
    imputed_data_x, MSE = gain(miss_data_x, gain_parameters)

    if str(imputed_data_x[0][0]) != 'nan':
        # np.savetxt("data/adjusted/" + data_name + "missing.csv", miss_data_x, delimiter=",")
        np.savetxt("data/adjusted/" + data_name + "imputed.csv", imputed_data_x, delimiter=",")

    return MSE


def clean(data_name):
    data = pd.read_csv('data/adjusted/' + data_name + 'imputed.csv',
                       names=['GENDER_SRC_DESC', 'MARITAL_STATUS_AT_DX_DESC', 'ETHNICITY_SRC_DESC',
                              'RACE_CR_SRC_DESC_1', 'HISTOLOGY_CD', 'PRIMARY_SITE_CD',
                              'PRIMARY_SITE_GROUP_DESC', 'PRIMARY_SITE_REGION_DESC',
                              'METS_AT_DX_DISTANT_LYMPH_NODES_DESC', 'METS_AT_DX_OTHER_DESC',
                              'SUMMARY_OF_RX_1ST_COURSE_DESC', 'SURGERY_RADIATION_SEQ_NUM',
                              'SYSTEMIC_RX_SURGERY_SEQ_NUM', 'DIAGNOSTIC_CONFIRMATION_METHOD_DESC',
                              'TUMOR_BEHAVIOR_DESC', 'VITAL_STATUS', 'Cessation Meds',
                              'Cessation Meds Past 90 Days', 'CRA_SMOKE_5PK_EVER',
                              'CRA_SMOKE_HOME_EXPOSED', 'CRA_SMOKE_WORK_EXPOSED', 'CRA_TOBACCO_SNUFF',
                              'AGE_AT_DIAGNOSIS_NUM', 'REGIONAL_NODES_EXAMINED',
                              'REGIONAL_NODES_POSITIVE', 'SURVIVAL_TIME_IN_MONTHS',
                              'PreTreatment_Smoke_Status', 'Tumor_size', 'Quit_Status_Any_time',
                              'CRA_SMOKE_AGE', 'CRA_SMOKE_TOTALYRS'])
    data = data.astype('int64')
    data.to_csv('data/adjusted/' + data_name + 'imputed.csv', index=False)
    return data


def impute(BaseEstimator, TransformerMixin):
    class impute(BaseEstimator, TransformerMixin):
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X.fillna('nan')
            X.to_csv('data/temp.csv')
            mse = imputer('temp')
            while str(mse) == 'nan':
                mse = imputer('temp')
            X = pd.read_csv('data/adjusted/tempimputed.csv',
                            names=['GENDER_SRC_DESC', 'MARITAL_STATUS_AT_DX_DESC', 'ETHNICITY_SRC_DESC',
                                   'RACE_CR_SRC_DESC_1', 'HISTOLOGY_CD', 'PRIMARY_SITE_CD',
                                   'PRIMARY_SITE_GROUP_DESC', 'PRIMARY_SITE_REGION_DESC',
                                   'METS_AT_DX_DISTANT_LYMPH_NODES_DESC', 'METS_AT_DX_OTHER_DESC',
                                   'SUMMARY_OF_RX_1ST_COURSE_DESC', 'SURGERY_RADIATION_SEQ_NUM',
                                   'SYSTEMIC_RX_SURGERY_SEQ_NUM', 'DIAGNOSTIC_CONFIRMATION_METHOD_DESC',
                                   'TUMOR_BEHAVIOR_DESC', 'VITAL_STATUS', 'Cessation Meds',
                                   'Cessation Meds Past 90 Days', 'CRA_SMOKE_5PK_EVER',
                                   'CRA_SMOKE_HOME_EXPOSED', 'CRA_SMOKE_WORK_EXPOSED', 'CRA_TOBACCO_SNUFF',
                                   'AGE_AT_DIAGNOSIS_NUM', 'REGIONAL_NODES_EXAMINED',
                                   'REGIONAL_NODES_POSITIVE', 'SURVIVAL_TIME_IN_MONTHS',
                                   'PreTreatment_Smoke_Status', 'Tumor_size', 'Quit_Status_Any_time',
                                   'CRA_SMOKE_AGE', 'CRA_SMOKE_TOTALYRS'])
            X = X.astype('int64')
            return X


# if __name__ == '__main__':

df = pd.read_csv('data/converted.csv')
X = df.drop(['Quit_FU'], axis=1)
y = df['Quit_FU']
cols = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

space = {
    'n_d': hp.quniform('n_d', 8, 64, 1),
    'n_a': hp.quniform('n_a', 8, 64, 1),
    'n_steps': hp.quniform('n_steps', 3, 10, 1),
    'n_independent': hp.quniform('n_independent', 1, 5, 1),
    'gamma': hp.uniform('gamma', 1, 2),
    'n_shared': hp.quniform('n_shared', 1, 5, 1),
    'momentum': hp.loguniform('momentum', -2, -0.398),
    'lambda_sparse': hp.loguniform('lambda_sparse', -5, -3)}


def objective(params):
    params['n_d'] = int(params['n_d'])
    params['n_a'] = int(params['n_a'])
    params['n_steps'] = int(params['n_steps'])
    params['n_independent'] = int(params['n_independent'])
    params['n_shared'] = int(params['n_shared'])
    pipeline = Pipeline([
        ("imputer", impute()),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2)),
        ("classifier", TabNetClassifier(**params, verbose=0))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scores = cross_val_score(pipeline, X_train.values, y_train.values, cv=cv, scoring='accuracy', n_jobs=-1)
    avg_score = np.mean(scores)
    return {'loss': -avg_score, 'status': STATUS_OK}


trials = Trials()

best_params = fmin(objective, space, algo=tpe.suggest, max_evals=10, trials=trials)
best_params['n_d'] = int(best_params['n_d'])
best_params['n_a'] = int(best_params['n_a'])
best_params['n_steps'] = int(best_params['n_steps'])
best_params['n_independent'] = int(best_params['n_independent'])
best_params['n_shared'] = int(best_params['n_shared'])
print("Best set of parameters: {}".format(best_params))

pipe = Pipeline([
        ("imputer", impute()),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2)),
        ("classifier", TabNetClassifier(**best_params))
    ])

pipe.fit(X_train.values, y_train.values)
predictions = pipe.predict(X_test.values)
print(y_test.value_counts())
print(accuracy_score(y_test, predictions))
print(roc_auc_score(y_test, predictions))
# accuracy: 0.5667517006802721
# roc_auc: 0.5659608702040451
# majority class percentage: 1213/(1213+1139)=0.5157