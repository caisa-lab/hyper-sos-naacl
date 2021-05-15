import argparse
import pickle
import os
import numpy as np
from sklearn import preprocessing
import warnings
import networkx as nx

warnings.filterwarnings('ignore')


def load_data(data_dir):
    with open(os.path.join(data_dir, 'df_hist.pkl'), "rb") as f:
        df_his = pickle.load(f)

    return df_his


def hawkes(features, time, epsilon, beta):
    decay = epsilon * np.exp(-beta * time)
    pos = np.greater(features, 0)
    hawkes_out = features + \
        features * pos.astype(np.float32) * np.expand_dims(decay, 1)
    return np.sum(hawkes_out, 0)


def get_timestamp(x):
    timestamp = []
    for t in x:
        timestamp.append((x[0] - t).days)

    return np.asarray(timestamp, dtype=np.float32)


def scale(x):
    return preprocessing.scale(hawkes(x[1], get_timestamp(x[0]), 0.05, 0.001))


def read_pickle(file):
    with open(file, 'rb') as f:
        data = pickle.load(f)

    return data


def apply_heat(config):
    data_dir = config['datapath']

    df_his = load_data(data_dir)

    df_his['enc'] = df_his[['hist_dates', 'enc']].progress_apply(scale, axis=1)

    df_concat = df_his[['lemmas', 'enc']]

    df = read_pickle(os.path.join(data_dir, 'df_complete.pkl'))
    g = nx.read_gpickle(os.path.join(data_dir, 'twitter_graph.p'))

    df_enc = df.merge(df_concat, on='lemmas', how='inner',
                      suffixes=("_tweet", "_hist")).drop_duplicates('tweet_id')

    node_list = sorted(g.nodes)

    user_enc = np.zeros((len(node_list), 768))

    non_dup_iterator = df_enc[['user_id', 'enc_hist']].drop_duplicates('user_id').iterrows()

    for row in non_dup_iterator:
        row = row[1]
        user_enc[node_list.index(int(row['user_id']))] = np.asarray(row['enc_hist'])

    for row in df_enc[['tweet_id', 'enc_tweet']].iterrows():
        row = row[1]
        user_enc[node_list.index(int(row['tweet_id']))] = np.asarray(row['enc_tweet'])

    np.save(os.path.join(data_dir, 'node_hawkes_enc.npy'), user_enc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath',
                        default="data/twitter/",
                        help="path to data")

    args = parser.parse_args()

    apply_heat(args.__dict__)
