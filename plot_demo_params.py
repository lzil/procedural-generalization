import os
import sys

import csv

import pdb

import numpy as np
import matplotlib.pyplot as plt

from correlations import retain_row


def main():
    demo_csv = 'trex/demos/demo_infos.csv'

    # use sequential constraint
    constraints = {
        'env_name': 'starpilot',
        'sequential': '200000000',
        'mode': 'easy'
    }

    with open(demo_csv, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        lens = []
        rets = []
        for row in reader:
            if not retain_row(row, constraints):
                continue
            lens.append(float(row['length']))
            rets.append(float(row['return']))

    lens = np.log(lens)
    rets = np.log(rets)

    seq2 = (lens, rets)

    # same as above, but don't use sequential
    constraints = {
        'env_name': 'starpilot',
        'sequential': '0',
        'mode': 'easy'
    }

    with open(demo_csv, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        lens = []
        rets = []
        for row in reader:
            if not retain_row(row, constraints):
                continue
            lens.append(float(row['length']))
            rets.append(float(row['return']))

    lens = np.log(lens)
    rets = np.log(rets)

    seq1 = (lens, rets)

    # plot it all

    plt.scatter(seq1[0], seq1[1], marker='.', color='orangered', alpha=0.3, label='random')
    plt.scatter(seq2[0], seq2[1], marker='.', color='forestgreen', alpha=0.3, label='sequential')

    # plt.yscale('log')
    # plt.xscale('log')

    plt.title('length vs return of demonstration', fontdict={'fontsize':20, 'fontweight':'bold'})
    plt.legend()
    plt.xlabel('log length')
    plt.ylabel('log return')
    plt.show()

    


if __name__ == '__main__':
    main()