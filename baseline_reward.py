import numpy as np
import pandas as pd


import argparse
import os
import pdb

from train_reward import get_demo, create_training_data



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('demo_csv')

    args = parser.parse_args()
    args.min_snippet_length = 20
    args.max_snippet_length = 100

    infos = pd.read_csv(args.demo_csv)

    infos = infos[infos.set_name == 'test']

    print(infos.shape)

    pdb.set_trace()

    test_dems = []
    for seed in infos.demo_id:
        for folder in args.demo_folder:
            fpath = os.path.join(folder, seed + '.demo')
            if os.path.isfile(fpath):
                test_dems.append(get_demo(fpath))
                break

    test_set, _ = create_dataset(
        dems = test_dems,
        num_snippets = 1000,
        min_snippet_length = args.min_snippet_length,
        max_snippet_length = args.max_snippet_length,
        validation = False,
        verbose = False
    )



if __name__ == '__main__':
    main()