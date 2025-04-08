# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import argparse
import random
import glob
from collections import defaultdict
import json

def main(args):
    random.seed(args.seed)
    ratio = [int(r) for r in args.split_ratio.split(":")]
    ratio = [r/sum(ratio) for r in ratio]

    dat = defaultdict(list)
    for lyrics_path in sorted(glob.glob(args.lyrics_data.rstrip("/") + "/*.joblib")):
        song_id = lyrics_path.split("/")[-1].split(".")[0]
        dat[song_id].append(lyrics_path)

    for lyrics_path in sorted(glob.glob(args.lyrics_data4clustering.rstrip("/") + "/*.joblib")):
        song_id = lyrics_path.split("/")[-1].split(".")[0]
        dat[song_id].append(lyrics_path)

    for motion_path in sorted(glob.glob(args.motion_data.rstrip("/") + "/*.joblib")):
        song_id = motion_path.split("/")[-1].split(".")[0]
        dat[song_id].append(motion_path)

    path_list = []
    for song_id, path_set in sorted(dat.items()):
        if len(path_set) == 3:
            path_list.append(path_set)

    random.shuffle(path_list)
    data_size = len(path_list)
    th_dev = int(data_size*ratio[0])
    th_test = int(data_size*(ratio[0]+ratio[1]))

    os.makedirs("./instance", exist_ok=True)
    fp = open("./instance/train.jsonl", "w")
    for path_set in path_list[:th_dev]:
        song_id = path_set[0].split("/")[-1].split(".")[0]
        fp.write(json.dumps({"song_id":song_id, 
                             "lyrics":os.path.abspath(path_set[0]), 
                             "lyrics4clustering":os.path.abspath(path_set[1]), 
                             "motion":os.path.abspath(path_set[2])}) + "\n")
    fp.close()

    fp = open("./instance/dev.jsonl", "w")
    for path_set in path_list[th_dev:th_test]:
        song_id = path_set[0].split("/")[-1].split(".")[0]
        fp.write(json.dumps({"song_id":song_id, 
                             "lyrics":os.path.abspath(path_set[0]), 
                             "lyrics4clustering":os.path.abspath(path_set[1]), 
                             "motion":os.path.abspath(path_set[2])}) + "\n")
    fp.close()

    fp = open("./instance/test.jsonl", "w")
    for path_set in path_list[th_test:]:
        song_id = path_set[0].split("/")[-1].split(".")[0]
        fp.write(json.dumps({"song_id":song_id, 
                             "lyrics":os.path.abspath(path_set[0]), 
                             "lyrics4clustering":os.path.abspath(path_set[1]), 
                             "motion":os.path.abspath(path_set[2])}) + "\n")
    fp.close()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="seed value")
    parser.add_argument("-lyrics_data", "--lyrics_data", dest="lyrics_data", default="./lyrics_data", type=str, help="dirctory path for lyrics data")
    parser.add_argument("-lyrics_data4clustering", "--lyrics_data4clustering", dest="lyrics_data4clustering", 
                        default="./lyrics_data4clustering", type=str, help="dirctory path for lyrics data for k-means clustering")
    parser.add_argument("-motion_data", "--motion_data", dest="motion_data", default="./motion_data", type=str, help="dirctory path for motion data")
    parser.add_argument("-split_ratio", "--split_ratio", dest="split_ratio", default="8:1:1", type=str, help="train/dev/test ratio (default: 8:1:1)")
    args = parser.parse_args()
    main(args)

