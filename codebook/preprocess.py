# -*- coding: utf-8 -*-
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import argparse
import json
import joblib
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import networkx as nx

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)
model.eval()

pad_token_id = tokenizer.pad_token_id
input_ids = torch.tensor([[pad_token_id]]).to(device)
with torch.no_grad():
    outputs = model(input_ids)
    last_hidden_states = outputs.last_hidden_state
pad_token_vector = last_hidden_states[0, 0, :].cpu().numpy()

def main(args):
    os.makedirs("./data", exist_ok=True)
    """ Preprocess lyrics data """
    word_vectors4train = []
    sentence_vectors4train = []
    for strm in open(args.train_path, "r"):
        js_dat = json.loads(strm.strip())
        lyrics_path = js_dat["lyrics4clustering"]
        lyrics_dat = joblib.load(lyrics_path)
        """ Lyrics vector unification """
        sentence2vecs = {}
        for dat in lyrics_dat:
            sentence = " ".join(dat["words"])
            sentence2vecs[sentence] = dat
        # One additional PAD token vector per song.
        sentence2vecs["[PAD]"] = {"words":["[PAD]"], "sentence_vector": pad_token_vector, "word_vectors":[pad_token_vector]}

        for k, v in sentence2vecs.items():
            sentence_vector = v["sentence_vector"]
            sentence_vectors4train.append(sentence_vector[:])
            for word_vector in v["word_vectors"]:
                word_vectors4train.append(word_vector[:])
    word_vectors4train = np.array(word_vectors4train)
    sentence_vectors4train = np.array(sentence_vectors4train)
    joblib.dump(word_vectors4train, "./data/word_vector4train.joblib", compress=3)
    joblib.dump(sentence_vectors4train, "./data/sentence_vector4train.joblib", compress=3)

    """ Preprocess motion data """
    motion_vectors = joblib.load(args.train_motion)
    m_data = []
    for motion_dat in motion_vectors:
        motion_vec = motion_dat["motion_vector"]
        m_data.append(motion_vec)
    m_data = np.concatenate(m_data)
    threshold = 0.99
    """ Motion vector unification """
    n_data = m_data.shape[0]
    dim = m_data.shape[1]
    faiss.normalize_L2(m_data)
    res = faiss.StandardGpuResources()
    index = faiss.IndexFlatIP(dim)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)  
    gpu_index.add(m_data)  
    k = 1000
    D, I = gpu_index.search(m_data, k)
    edges = []
    G = nx.Graph()
    G.add_nodes_from(range(n_data))  
    for i in range(n_data):
        for j in range(k):  
            if D[i, j] >= threshold:
                if i != I[i, j]:
                    edges.append((i, I[i, j]))
            else:
                break
    G.add_edges_from(edges)
    unique_features_indices = [min(c) for c in nx.connected_components(G)]
    unique_motion_vectors = m_data[unique_features_indices]
    joblib.dump(unique_motion_vectors, "./data/motion_vector4train.joblib", compress=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_path", "--train_path", dest="train_path", default="../data/instance/train.jsonl", type=str, help="training data jsonl path")
    parser.add_argument("-train_motion", "--train_motion", dest="train_motion", default="../model/checkpoint/model_skeletal1_aft1_adamw_lr1e-05/motion_vector_train.joblib", type=str, help="training motion vector joblib file")
    args = parser.parse_args()
    main(args)

