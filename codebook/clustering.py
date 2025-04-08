# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import argparse
import joblib
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def main(args):
    np.random.seed(args.seed)

    sys.stderr.write("k-means clustering... (word vectors)\n")
    os.makedirs("./model", exist_ok=True)
    X = joblib.load(args.word_vec)
    np.random.shuffle(X)
    kmeans = MiniBatchKMeans(n_clusters=args.LK, random_state=0, n_init="auto", compute_labels=False, batch_size=5000, verbose=0).fit(X)
    joblib.dump(kmeans, "./model/kmeans_word_%04d.joblib"%(args.LK), compress=3)

    sys.stderr.write("k-means clustering... (sentence vectors)\n")
    X = joblib.load(args.sentence_vec)
    np.random.shuffle(X)
    kmeans = MiniBatchKMeans(n_clusters=args.LK, random_state=0, n_init="auto", compute_labels=False, batch_size=5000, verbose=0).fit(X)
    joblib.dump(kmeans, "./model/kmeans_sentence_%04d.joblib"%(args.LK), compress=3)

    sys.stderr.write("k-means clustering... (motion vectors)\n")
    X = joblib.load(args.motion_vec)
    np.random.shuffle(X)
    kmeans = MiniBatchKMeans(n_clusters=args.MK, random_state=0, n_init="auto", compute_labels=False, batch_size=5000, verbose=0).fit(X)
    joblib.dump(kmeans, "./model/kmeans_motion_%04d.joblib"%(args.MK), compress=3)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="seed value")
    parser.add_argument("-LK", "--LK", dest="LK", default=500, type=int, help="number of lyrics codebooks")
    parser.add_argument("-MK", "--MK", dest="MK", default=500, type=int, help="number of motion codebooks")
    parser.add_argument("-word_vec", "--word_vec", dest="word_vec", default="./data/word_vector4train.joblib", type=str, help="path of word vector file")
    parser.add_argument("-sentence_vec", "--sentence_vec", dest="sentence_vec", default="./data/sentence_vector4train.joblib", type=str, help="path of sentence vector file")
    parser.add_argument("-motion_vec", "--motion_vec", dest="motion_vec", default="./data/motion_vector4train.joblib", type=str, help="path of motion vector file")

    args = parser.parse_args()
    main(args)

