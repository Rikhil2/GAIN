from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np

from data_loader import data_loader
from gain import gain
from utils import rmse_loss


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

    return MSE  # imputed_data_x[0][0]


if __name__ == '__main__':

    # Calls main function

    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--batch_size',
    #     help='the number of samples in mini-batch',
    #     default=128,
    #     type=int)
    # parser.add_argument(
    #     '--hint_rate',
    #     help='hint probability',
    #     default=0.9,
    #     type=float)
    # parser.add_argument(
    #     '--alpha',
    #     help='hyperparameter',
    #     default=100,
    #     type=float)
    # parser.add_argument(
    #     '--iterations',
    #     help='number of training iterations',
    #     default=1,
    #     type=int)
    #
    # args = parser.parse_args()
    # num = main(args)
    # print(num)
    best_mse = float('inf')
    best_alpha = 0
    for i in [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9]:
        print(i)
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
            '--alpha',  # -> multiplier on how much MSE_loss affects generator loss
            help='hyperparameter',  # larger values of alpha will increase generator loss
            # more generator loss -> larger gradients
            default=i,  # 100
            type=float)
        parser.add_argument(
            '--iterations',
            help='number of training iterations',
            default=200,  # 10000
            type=int)
        args = parser.parse_args()
        mse = main(args)
        print(mse)
        if mse < best_mse:
            best_mse = mse
            best_alpha = i
    print(best_alpha)
    print(best_mse)
    # boo = True
    # counter = 1
    # while boo:
    #     counter *= 2
    #     print(counter)
    #     parser = argparse.ArgumentParser()
    #     parser.add_argument(
    #         '--batch_size',
    #         help='the number of samples in mini-batch',
    #         default=1024,
    #         type=int)
    #     parser.add_argument(
    #         '--hint_rate',
    #         help='hint probability',
    #         default=0.9,
    #         type=float)
    #     parser.add_argument(
    #         '--alpha',
    #         help='hyperparameter',
    #         default=100,
    #         type=float)
    #     parser.add_argument(
    #         '--iterations',
    #         help='number of training interations',
    #         default=counter,
    #         type=int)
    #
    #     args = parser.parse_args()
    #     num = main(args)
    #     if str(num) == 'nan':
    #         boo = False
