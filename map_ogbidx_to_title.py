import pandas as pd
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ogbidx', type=int, default=0)
    args = parser.parse_args()

    idx_title = pd.read_csv('/data/yuwen/paperinfo/idx_title.tsv', sep='\t')
    idx = idx_title.iloc[:, 0].tolist()

    for i, id in enumerate(idx):
        if id == args.ogbidx:
            print(idx_title.iat[i, 1])
            break
