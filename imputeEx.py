from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from gain import gain


def main(args):
    gain_parameters = {'batch_size': args.batch_size,
                       'hint_rate': args.hint_rate,
                       'alpha': args.alpha,
                       'iterations': args.iterations}

    # Load data
    miss_data_x = np.loadtxt('data/' + 'converted' + '.csv', delimiter=",", skiprows=1)

    # Impute missing data
    imputed_data_x, MSE = gain(miss_data_x, gain_parameters)

    if str(imputed_data_x[0][0]) != 'nan':
        np.savetxt("data/adjusted/missing.csv", miss_data_x, delimiter=",")
        np.savetxt("data/adjusted/imputed.csv", imputed_data_x, delimiter=",")

    return MSE


if __name__ == '__main__':
    # Calls main function

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=128,
        type=int)
    parser.add_argument(
        '--hint_rate',
        help='hint probability',
        default=0.9,
        type=float)
    parser.add_argument(
        '--alpha',
        help='hyperparameter',
        default=1e4,
        type=float)
    parser.add_argument(
        '--iterations',
        help='number of training iterations',
        default=10000,
        type=int)
    args = parser.parse_args()
    mse = main(args)
    while str(mse) == 'nan':
        mse = main(args)
    print(mse)