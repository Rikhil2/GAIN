from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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

    X_train = pd.DataFrame(imputed_data, columns=X_test.columns)
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


# Usage
df = pd.read_csv('data/converted.csv')
X = df.drop(['Quit_FU'], axis=1)
y = df['Quit_FU']
X = X.fillna(np.nan)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gain_parameters = {'batch_size': 128, 'hint_rate': 0.9, 'alpha': 1e4, 'iterations': 10000}

# Train the imputer
sess, G_sample, X, M, norm_parameters, X_train_imputed = train_imputer(X_train, gain_parameters)

# Impute the test data
X_test_imputed = impute_test_data(sess, G_sample, X, M, norm_parameters, X_test)
# X_train_imputed.to_csv('X_train_imputed.csv', index=False)
# X_test_imputed.to_csv('X_test_imputed.csv', index=False)

alpha = [4.5, 5, 5.5]
gamma = [0]
reg_lambda = [2.5, 3, 3.5]
learning_rate = [0.05, 0.1, 0.15]

classifier = XGBClassifier()

classifier_gscv = GridSearchCV(
    estimator=classifier,
    param_grid={
        'learning_rate': learning_rate,
        'gamma': gamma,
        'lambda': reg_lambda,
        'alpha': alpha
    },
    scoring='accuracy',
    cv=5
)

classifier_gscv.fit(X_train_imputed, y_train)
print(classifier_gscv.best_params_)

# Create a new classifier instance with the best parameters
clf = XGBClassifier(**classifier_gscv.best_params_)

clf.fit(X_train_imputed, y_train)

preds = clf.predict(X_test_imputed)

print(accuracy_score(y_test, preds))
print(roc_auc_score(y_test, preds))
