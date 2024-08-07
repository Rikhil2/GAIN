from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

from gain import gain
from sklearn.model_selection import train_test_split


def imputer(batch_size, hint_rate, alpha, iterations, data_name):
    gain_parameters = {'batch_size': batch_size,
                       'hint_rate': hint_rate,
                       'alpha': alpha,
                       'iterations': iterations}

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


# if __name__ == '__main__':
batch_size = 128
hint_rate = 0.9
alpha = 1e4
iterations = 10000

df = pd.read_csv('data/converted.csv')
X = df.drop(['Quit_FU'], axis=1)
y = df['Quit_FU']

cols = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

X_train = X_train.fillna('nan')
X_test = X_test.fillna('nan')

X_train.to_csv('data/X_train.csv')
X_test.to_csv('data/X_test.csv')

mse = imputer(batch_size, hint_rate, alpha, iterations, 'X_train')
while str(mse) == 'nan':
    mse = imputer(batch_size, hint_rate, alpha, iterations, 'X_train')
print(mse)
clean('X_train')

mse = imputer(batch_size, hint_rate, alpha, iterations, 'X_test')
while str(mse) == 'nan':
    mse = imputer(batch_size, hint_rate, alpha, iterations, 'X_test')
print(mse)
clean('X_test')

X_train = pd.read_csv('data/adjusted/X_trainimputed.csv')
X_test = pd.read_csv('data/adjusted/X_testimputed.csv')

eta = np.arange(0.3, 0.8, 0.1)
gamma = [0, 0.5, 1, 10, 100, 1000]
max_depth = [4, 5, 6, 7, 8]
min_child_weight = [0.1, 1, 10]
max_delta_step = np.arange(5, 8, 1)
subsample = np.arange(0.5, 0.8, 0.1)
reg_lambda = [0.1, 1, 10]
alpha = [0.1, 1, 10]

classifier = XGBClassifier()  # eta=0.3, gamma=0, max_depth=6, min_child_weight=1, max_delta_step=0, subsample=1, reg_lambda=1, alpha=0, scale_pos_weight=1

classifier_gscv = GridSearchCV(
    estimator=classifier,
    param_grid={
        'eta': eta,
        'gamma': gamma,
        'min_child_weight': min_child_weight,
        'max_delta_step': max_delta_step,
        'subsample': subsample,
        'reg_lambda': reg_lambda,
        'alpha': alpha
    },
    scoring='accuracy',
    cv=5
)

classifier_gscv.fit(X_train, y_train)
clf = XGBClassifier(classifier_gscv.best_params_)

clf.fit(X_train, y_train)

preds = clf.predict(X_test)

print(accuracy_score(y_test, preds))#