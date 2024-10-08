from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from hyperopt import fmin, Trials, STATUS_OK, tpe, hp
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import make_scorer, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

import tensorflow.compat.v1 as tf

from utils import normalization, sample_batch_index, uniform_sampler, binary_sampler, rounding, renormalization, \
    xavier_init

tf.disable_v2_behavior()

def train_imputer(X_train, gain_parameters):
    miss_data_x = X_train.to_numpy()
    data_m = 1 - np.isnan(miss_data_x)

    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']

    no, dim = miss_data_x.shape
    h_dim = int(dim)

    norm_data, norm_parameters = normalization(miss_data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    X = tf.placeholder(tf.float32, shape=[None, dim])
    M = tf.placeholder(tf.float32, shape=[None, dim])
    H = tf.placeholder(tf.float32, shape=[None, dim])

    D_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    D_W3 = tf.Variable(xavier_init([h_dim, dim]))
    D_b3 = tf.Variable(tf.zeros(shape=[dim]))
    theta_D = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    G_W1 = tf.Variable(xavier_init([dim * 2, h_dim]))
    G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
    G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))
    G_W3 = tf.Variable(xavier_init([h_dim, dim]))
    G_b3 = tf.Variable(tf.zeros(shape=[dim]))
    theta_G = [G_W1, G_W2, G_W3, G_b1, G_b2, G_b3]

    def generator(x, m):
        inputs = tf.concat(values=[x, m], axis=1)
        G_h1 = tf.nn.relu(tf.matmul(inputs, G_W1) + G_b1)
        G_h2 = tf.nn.relu(tf.matmul(G_h1, G_W2) + G_b2)
        G_prob = tf.nn.sigmoid(tf.matmul(G_h2, G_W3) + G_b3)
        return G_prob

    def discriminator(x, h):
        inputs = tf.concat(values=[x, h], axis=1)
        D_h1 = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
        D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
        D_logit = tf.matmul(D_h2, D_W3) + D_b3
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob

    G_sample = generator(X, M)
    Hat_X = X * M + G_sample * (1 - M)
    D_prob = discriminator(Hat_X, H)

    D_loss_temp = -tf.reduce_mean(M * tf.log(D_prob + 1e-8) + (1 - M) * tf.log(1. - D_prob + 1e-8))
    G_loss_temp = -tf.reduce_mean((1 - M) * tf.log(D_prob + 1e-8))
    MSE_loss = tf.reduce_mean((M * X - M * G_sample) ** 2) / tf.reduce_mean(M)

    D_loss = D_loss_temp
    G_loss = G_loss_temp + alpha * MSE_loss

    D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
    G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    best_loss = float('inf')
    patience_counter = 0

    for it in range(iterations):
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = norm_data_x[batch_idx, :]
        M_mb = data_m[batch_idx, :]
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        H_mb_temp = binary_sampler(hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

        _, D_loss_curr = sess.run([D_solver, D_loss_temp], feed_dict={M: M_mb, X: X_mb, H: H_mb})
        _, G_loss_curr, MSE_loss_curr = sess.run([G_solver, G_loss_temp, MSE_loss],
                                                 feed_dict={X: X_mb, M: M_mb, H: H_mb})

        if MSE_loss_curr < best_loss:
            best_loss = MSE_loss_curr
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"Early stopping on iteration {it}. Best loss: {best_loss}")
                break

    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]

    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data

    # Renormalization
    imputed_data = renormalization(imputed_data, norm_parameters)

    # Rounding
    imputed_data = rounding(imputed_data, miss_data_x)

    X_train = pd.DataFrame(imputed_data, columns=X_train.columns)
    return sess, G_sample, X, M, norm_parameters, X_train


def impute_test_data(sess, G_sample, X, M, norm_parameters, X_test):
    miss_data_x = X_test.to_numpy()
    data_m = 1 - np.isnan(miss_data_x)
    no, dim = miss_data_x.shape
    norm_data, _ = normalization(miss_data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    Z_mb = uniform_sampler(0, 0.01, no, dim)
    M_mb = data_m
    X_mb = norm_data_x
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb

    imputed_data = sess.run([G_sample], feed_dict={X: X_mb, M: M_mb})[0]
    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
    imputed_data = renormalization(imputed_data, norm_parameters)
    imputed_data = rounding(imputed_data, miss_data_x)

    return pd.DataFrame(imputed_data, columns=X_test.columns)


class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, gain_parameters):
        self.gain_parameters = gain_parameters
        self.sess = None
        self.G_sample = None
        self.Q = None
        self.M = None
        self.norm_parameters = None

    def fit(self, X, y=None):
        self.sess, self.G_sample, self.Q, self.M, self.norm_parameters, _ = train_imputer(X, self.gain_parameters)
        return self

    def transform(self, X):
        X_copy = X.copy()
        if X_copy.shape[0] < 4000:
            _, _, _, _, _, X_train_imputed = train_imputer(X_copy, self.gain_parameters)
            return X_train_imputed
        else:
            X_test_imputed = impute_test_data(self.sess, self.G_sample, self.Q, self.M, self.norm_parameters, X_copy)
            if str(X_test_imputed['GENDER_SRC_DESC'][0]) == 'nan':
                while str(X_test_imputed[0][0]) == 'nan':
                    X_test_imputed = impute_test_data(self.sess, self.G_sample, self.Q, self.M, self.norm_parameters, X_copy)
            # print(X_test_imputed['GENDER_SRC_DESC'][0])
            return X_test_imputed


# Load the data
df = pd.read_csv('data/converted.csv')
X = df.drop(['Quit_FU'], axis=1)
y = df['Quit_FU']
X = X.fillna(np.nan)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gain_parameters = {'batch_size': 128, 'hint_rate': 0.9, 'alpha': 1e4, 'iterations': 10000}

# Train the imputer
# sess, G_sample, X, M, norm_parameters, X_train_imputed = train_imputer(X_train_val, gain_parameters)
# X_test_imputed = impute_test_data(sess, G_sample, X, M, norm_parameters, X_test)


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
        ("imputer", CustomImputer(gain_parameters)),
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2)),
        ("classifier", TabNetClassifier(**params))
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True)
    scorer = make_scorer(mean_squared_error, greater_is_better=False)
    scores = cross_val_score(pipeline, X_train_val, y_train_val, cv=cv, scoring=scorer, n_jobs=-1)
    avg_score = np.mean(scores)
    return {'loss': avg_score, 'status': STATUS_OK}


trials = Trials()

best_params = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=trials)
best_params['n_d'] = int(best_params['n_d'])
best_params['n_a'] = int(best_params['n_a'])
best_params['n_steps'] = int(best_params['n_steps'])
best_params['n_independent'] = int(best_params['n_independent'])
best_params['n_shared'] = int(best_params['n_shared'])
print("Best set of parameters: {}".format(best_params))

pipe = Pipeline([
    ("imputer", CustomImputer(gain_parameters)),
    ("scaler", StandardScaler()),
    ("poly", PolynomialFeatures(degree=2)),
    ("classifier", TabNetClassifier(**best_params))
])

pipe.fit(X_train_val, y_train_val)
predictions = pipe.predict(X_test)
print(y_test.value_counts())
print(accuracy_score(y_test, predictions))
print(roc_auc_score(y_test, predictions))
# accuracy:
# roc_auc:
# majority class percentage:
