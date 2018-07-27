#!/usr/bin/env python
#
# 1.0-predict-histogram-clustering.py
#
# Author: David Riser
# Date:   July 27, 2018
#
# A fast histogram based clustering method is used
# to provide a baseline solution to the tracking
# problem.
#
# This algorithm uses the best features
# from previous works and is written for
# speed not for generalization.

import numpy as np
import pandas as pd
import time

from contextlib import contextmanager
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
from trackml.dataset import load_event, load_dataset
from trackml.score import score_event

class Clusterer(object):
    def __init__(self, n1=100, n2=100, n3=100, n4=100):
        ''' This method assumes the four features
        are scaled between 0 and 1. '''
        self.labels = None
        self.bins1 = np.linspace(0, 1, n1)
        self.bins2 = np.linspace(0, 1, n2)
        self.bins3 = np.linspace(0, 1, n3)
        self.bins4 = np.linspace(0, 1, n4)
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.n4 = n4

    def predict(self, X):
        idx1, idx2, idx3, idx4 = self.digitize_axes(X)
        self.assign_labels(idx1, idx2, idx3, idx4)
        return self.labels

    def digitize_axes(self, X):
        idx1 = np.digitize(X[:, 0], bins=self.bins1)
        idx2 = np.digitize(X[:, 1], bins=self.bins2)
        idx3 = np.digitize(X[:, 2], bins=self.bins3)
        idx4 = np.digitize(X[:, 3], bins=self.bins4)
        return idx1, idx2, idx3, idx4

    def assign_labels(self, idx1, idx2, idx3, idx4):
        self.labels = idx1 + \
            self.n1 * idx2 + \
            self.n1 * self.n2 * idx3 + \
            self.n1 * self.n2 * self.n3 * idx4

        self.labels -= np.min(self.labels)

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

def create_one_event_submission(event_id, hits, labels):
    sub_data = np.column_stack(([event_id]*len(hits), hits.hit_id.values, labels))
    submission = pd.DataFrame(data=sub_data, columns=['event_id', 'hit_id', 'track_id']).astype(int)
    return submission

def score_event_fast(truth, submission):
    ''' Score function by Kaggle user CPMP. '''
    truth = truth[['hit_id', 'particle_id', 'weight']].merge(submission, how='left', on='hit_id')
    df = truth.groupby(['track_id', 'particle_id']).hit_id.count().to_frame('count_both').reset_index(\
)
    truth = truth.merge(df, how='left', on=['track_id', 'particle_id'])

    df1 = df.groupby(['particle_id']).count_both.sum().to_frame('count_particle').reset_index()
    truth = truth.merge(df1, how='left', on='particle_id')
    df1 = df.groupby(['track_id']).count_both.sum().to_frame('count_track').reset_index()
    truth = truth.merge(df1, how='left', on='track_id')
    truth.count_both *= 2
    score = truth[(truth.count_both > truth.count_particle) & (truth.count_both > truth.count_track)].\
weight.sum()
    return score

def test_events():
    ''' Process a few events to demonstrate the method
    before using it on the entire testing dataset.
    '''
    events        = range(1000, 1005)
    path_to_train = '../../data/raw/train_100_events/'
    parts_to_load = ['hits', 'truth']

    model = Clusterer(350, 2, 40, 40)
    for event_id in events:

        with timer('Processing %s' % event_id):
            hits, truth = load_event(path_to_train+'event00000'+str(event_id), parts=parts_to_load)

            # Add the needed information.
            r = np.sqrt(hits['x'].values ** 2 + hits['y'].values ** 2)
            d = np.sqrt(hits['x'].values ** 2 + hits['y'].values ** 2 + hits['z'].values**2)
            hits['phi'] = np.arctan2(hits['y'].values, hits['x'].values)
            hits['sphi'] = np.sin(hits['phi'].values)
            hits['cphi'] = np.cos(hits['phi'].values)
            hits['zd'] = hits['z'].values / d
            hits['zr'] = hits['z'].values / r

            scaler = MinMaxScaler()
            X = scaler.fit_transform(hits[['zr', 'zd', 'sphi', 'cphi']])

            labels = model.predict(X)
            submission = create_one_event_submission(event_id, hits, labels)
            score = score_event_fast(truth, submission)
            print('score = %.4f' % score)

        with timer('DBSCAN'):
            dbs = DBSCAN(eps=0.008)
            labels = dbs.fit_predict(X)
            submission = create_one_event_submission(event_id, hits, labels)
            score = score_event_fast(truth, submission)
            print('score = %.4f' % score)

if __name__ == '__main__':
    test_events()
