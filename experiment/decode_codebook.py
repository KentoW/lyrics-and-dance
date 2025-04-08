# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import joblib
import argparse
import numpy as np
from collections import defaultdict

def main(args):
    kmeans_lyrics = joblib.load(args.lyrics_model)
    LK = int(args.motion_model.split("_")[-1].split(".")[0])
    if "sentence" in args.lyrics_model.split("/")[-1]:
        lyrics_type = "sentence"
    elif "word" in args.lyrics_model.split("/")[-1]:
        lyrics_type = "word"
    else:
        sys.stderr("ERROR: The filename of the k-means model of the lyrics does not contain 'sentence' or 'word'. Please enter a k-means model for sentence or word.")
        exit(1)

    """ decode lyric codebooks """
    id2x = {}
    for strm in open(args.test_lyrics, "r"):
        js_dat = json.loads(strm.strip())
        song_id = js_dat["song_id"]
        lyrics_dat = joblib.load(js_dat["lyrics"])
        for bar_dat in lyrics_dat:
            start_frame = bar_dat[0]["frame"]
            vectors = []
            for line_dat in bar_dat:
                if lyrics_type == "sentence":
                    lyrics_vector = line_dat["sentence_vector"]
                else:
                    lyrics_vector = line_dat["word_vector"]
                vectors.append(lyrics_vector)
            X = kmeans_lyrics.predict(np.array(vectors))
            id2x[(song_id, start_frame)] = X

    """ decode motion codebooks """
    kmeans_motion = joblib.load(args.motion_model)
    MK = int(args.motion_model.split("_")[-1].split(".")[0])
    motion_vectors = joblib.load(args.test_motion)
    id2y = {}
    for motion_dat in motion_vectors:
        song_id = motion_dat["song_id"]
        start_frame = motion_dat["start_frame"]
        motion_vec = motion_dat["motion_vector"]
        Y = kmeans_motion.predict(motion_vec)
        Y += LK
        id2y[(song_id, start_frame)] = Y

    output_dir = "./result/%s2motion_LK-%04d_MK-%04d"%(lyrics_type, LK, MK)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump((id2y, id2x), output_dir+"/codebooks.joblib", compress=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-test_motion", "--test_motion", dest="test_motion", default="../model/checkpoint/model_skeletal1_aft1_adamw_lr1e-05/motion_vector_test.joblib", type=str, help="path of test motion vector")
    parser.add_argument("-test_lyrics", "--test_lyrics", dest="test_lyrics", default="../data/instance/test.jsonl", type=str, help="path of test lyrics json")

    parser.add_argument("-motion_model", "--motion_model", dest="motion_model", default="../codebook/model/kmeans_motion_0500.joblib", type=str, help="path of k-means model path for motion")
    parser.add_argument("-lyrics_model", "--lyrics_model", dest="lyrics_model", default="../codebook/model/kmeans_sentence_0500.joblib", type=str, help="path of k-means model path for lyrics")

    args = parser.parse_args()
    main(args)
