# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import argparse
import joblib
from collections import defaultdict
import math


def main(args):
    id2x, id2y = joblib.load(args.codebook)
    output_dir = "/".join(args.codebook.split("/")[:-1])

    """ Get NPMI """
    song2xy = defaultdict(lambda:set())
    song2x = defaultdict(lambda:set())
    song2y = defaultdict(lambda:set())
    for _id, X in id2x.items():
        song_id = _id[0]
        Y = id2y[_id]
        lyrics_cb = [str(x) for x in X]
        motion_cb = [str(y) for y in Y]
        for pair in zip(lyrics_cb, motion_cb):
            song2xy[song_id].add(pair)
            song2x[song_id].add(pair[0])
            song2y[song_id].add(pair[1])
    total_songs = len(song2xy)

    document_xy_freq = defaultdict(int)
    document_x_freq = defaultdict(int)
    document_y_freq = defaultdict(int)
    document_xy_sum = 0
    document_x_sum = 0
    document_y_sum = 0
    for song_id in song2xy.keys():
        for pair in song2xy[song_id]:
            document_xy_freq[pair] += 1
            document_xy_sum += 1
        for x in song2x[song_id]:
            document_x_freq[x] += 1
            document_x_sum += 1
        for y in song2y[song_id]:
            document_y_freq[y] += 1
            document_y_sum += 1

    npmi = {}
    overfit_pair = set()
    for pair, f_xy in document_xy_freq.items():
        f_x = document_x_freq[pair[0]]
        f_y = document_y_freq[pair[1]]
        p_xy = f_xy / total_songs
        p_x = f_x / total_songs
        p_y = f_y / total_songs
        document_pmi = math.log(p_xy / (p_x*p_y))
        npmi_value = document_pmi / -math.log(p_xy)
        npmi[pair] = npmi_value
        # Co-occurring document frequencies of 1 are ignored.
        if f_xy == 1 and npmi_value > 0:        
            overfit_pair.add(pair)

    fp = open(output_dir + "/npmi.txt", "w")
    for pair, cost in sorted(npmi.items(), key=lambda x:x[1], reverse=True):
        if pair in overfit_pair:
            pass
        else:
            fp.write("%s %s %s %s %s\n"%(pair[0], pair[1], cost, document_xy_freq[pair], total_songs))
    fp.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-codebook", "--codebook", dest="codebook", default="./result/sentence2motion_LK-0500_MK-0500/codebooks.joblib", type=str, help="path of codebook files")
    args = parser.parse_args()
    main(args)

