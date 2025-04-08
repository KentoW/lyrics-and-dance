# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import argparse
import joblib
import json
import subprocess
from collections import defaultdict



def calculate_rank(query_id, scores):
    R = defaultdict(lambda: set())
    for _id, score in scores.items():
        R[score].add(_id)
    rank = 0
    for score, ids in sorted(R.items(), reverse=True):
        rank += len(ids)
        if query_id in ids:
            return rank
    return rank

def main(args):
    output_dir = "/".join(args.codebook.split("/")[:-1])

    """ write cost file """
    npmi_path = output_dir + "/npmi.txt"
    cost_path = output_dir + "/cost.txt"
    positive_pair = set()
    fp = open(cost_path, "w")
    for strm in open(npmi_path, "r"):
        itm = strm.split(" ")
        tk = int(itm[0])
        mk = int(itm[1])
        npmi = float(itm[2])
        npmi_cost = 1 - ((1 + npmi) / 2)
        if npmi > 0.0:
            positive_pair.add((tk, mk))
        fp.write("%s %s %s\n"%(tk, mk, npmi_cost))
    fp.close()

    """ write data for dtw calculation """
    id2x, id2y = joblib.load(args.codebook)
    db_fp = open(output_dir + "/database.txt",  "w")
    qr_fp = open(output_dir + "/query.txt", "w")
    id_fp = open(output_dir + "/id.txt", "w")
    querys = []
    idx2length = []
    idx2id = []
    for _id, _X in id2x.items():
        X = [str(x) for x in _X]
        Y = [str(y) for y in id2y[_id]]
        qr_fp.write(" ".join(X) + "\n")
        db_fp.write(" ".join(Y) + "\n")
        id_fp.write("%s %s\n"%(_id[0], _id[1]))
        querys.append(" ".join(X))
        idx2length.append(len(X))
        idx2id.append(_id)
    db_fp.close()
    qr_fp.close()
    id_fp.close()

    """ calc DTW """
    R = {}
    RR = []
    total = len(querys)
    for idx, query in enumerate(querys):
        text_id = idx2id[idx]
        fp = open(output_dir+"/temp.txt", "w")
        fp.write(query)
        fp.close()

        qpath = output_dir+"/temp.txt"
        dbpath = output_dir+"/database.txt"
        costpath = output_dir+"/cost.txt"
        results = subprocess.run([args.dtw, qpath, dbpath, costpath, str(args.n_cpu)], capture_output=True, text=True)
        sims = {}
        for result in results.stdout.strip().split("\n"):
            _idx, dist = result.split(":")
            _idx = int(_idx)
            motion_id = idx2id[_idx]
            length = max(idx2length[_idx], idx2length[idx])
            score = 1-float(dist)/length
            sims[motion_id] = score
        rank = calculate_rank(text_id, sims)
        R[text_id] = (rank, sims)
        RR.append(1/rank)
        sys.stderr.write("\rMRR: %.5f [%s/%s]"%(sum(RR)/len(RR), idx+1, total))
    sys.stderr.write("\n")
    joblib.dump(R, output_dir+"/rank.joblib", compress=3)
    mrr = sum(RR)/len(RR)
    mrrfp = open(output_dir+"/mrr.json", "w")
    mrrfp.write(json.dumps({"mrr":mrr, "number_of_candidates":total}))
    mrrfp.close()
    sys.stderr.write("MRR: %.5f (rank positoin: %s)\n"%(mrr, int(1/mrr)))








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sample')
    parser.add_argument("-codebook", "--codebook", dest="codebook", default="./result/sentence2motion_LK-0500_MK-0500/codebooks.joblib", type=str, help="path of codebook files")
    parser.add_argument("-n_cpu", "--n_cpu", dest="n_cpu", default=4, type=int, help="number of CPUs used during parallel computation of DTW.")
    parser.add_argument("-dtw", "--dtw", dest="dtw", default="../dtw/dtw", type=str, help="Location of DTW commands.")
    args = parser.parse_args()
    main(args)
