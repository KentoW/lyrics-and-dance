# -*- coding: utf-8 -*-
import os
import sys
import argparse
import random
from collections import defaultdict
import nltk
nltk.download('words')
from nltk.corpus import words
from transformers import AutoTokenizer, AutoModel
import torch
import joblib

os.chdir(os.path.dirname(os.path.abspath(__file__)))

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

def generate_pseudo_text(word_list, num_sentences=5, sentence_length_range=(5, 15)):
    pseudo_text = []
    for _ in range(num_sentences):
        sentence_length = random.randint(*sentence_length_range)
        sentence = random.choices(word_list, k=sentence_length) 
        pseudo_text.append(sentence[:])
    return pseudo_text


def split_with_gaps_around(
    total_frames, num_segments, min_segment_length=5, gap_range=(1, 5)
):
    # Calculate max segment length based on total frames and number of segments
    max_segment_length = total_frames // num_segments

    # Step 1: Divide frames into segments without gaps
    remaining_frames = total_frames
    segment_lengths = []
    for i in range(num_segments):
        if i == num_segments - 1:
            # Ensure the last segment is within bounds
            segment_length = random.randint(min_segment_length, min(max_segment_length, remaining_frames))
        else:
            # Restrict segment length between min_segment_length and max_segment_length
            max_length = min(max_segment_length, remaining_frames - (min_segment_length * (num_segments - len(segment_lengths) - 1)))
            segment_length = random.randint(min_segment_length, max_length)
        
        segment_lengths.append(segment_length)
        remaining_frames -= segment_length

    # Step 2: Calculate segment start and end indices with gaps around them
    segments = []
    current_position = 0
    # Add gap before the first segment
    gap_before = random.randint(*gap_range)
    current_position += gap_before
    for length in segment_lengths:
        # Define the segment
        start = current_position
        end = start + length - 1
        # Ensure the segment does not exceed the total frame range
        if end >= total_frames:
            end = total_frames - 1
            length = end - start + 1  # Recalculate length if adjusted
        if start >= total_frames:
            break
        segments.append((start, end))
        current_position = end + 1
        # Add gap after the segment (even for the last one)
        gap_after = random.randint(*gap_range)
        current_position += gap_after
    return segments



def split_into_segments(N, W):
    if W <= 0:
        raise ValueError("W must be greater than zero.")
    if N <= 0:
        raise ValueError("N must be greater than zero.")
    # Calculate the number of segments
    num_segments = N // W
    remainder = N % W
    # Distribute the remainder across segments
    segments = []
    start = 0
    for i in range(num_segments):
        segment_width = W + (1 if i < remainder else 0)  # Add 1 to the first `remainder` segments
        end = start + segment_width - 1
        segments.append((start, end))
        start = end + 1
    # Handle the case where the last segment is shorter
    if start < N:
        segments.append((start, N - 1))
    return segments




def main(args):
    random.seed(args.seed)
    os.makedirs("./lyrics_data", exist_ok=True)
    os.makedirs("./lyrics_data4clustering", exist_ok=True)
    word_list = random.sample(words.words(), args.n_vocabs)
    for i in range(100):
        sys.stderr.write("\rGenerating pseudo lyrics [%03d/%03d]"%(i+1, 100))
        n_sentences = random.randint(20, 40)
        n_seconds = random.randint(200, 300)
        n_frames = 30*n_seconds
        # Pseudo-generation of timing for sentences to be sung
        segments = split_with_gaps_around(n_frames, n_sentences, min_segment_length=100, gap_range=(0, 10))
        # Pseudo-generation of sentences
        pseudo_sentences = generate_pseudo_text(word_list, n_sentences)

        bpm = random.randint(100, 130)
        seconds_per_bar = 4 * 60 / bpm
        n_frames_per_var = int(30 * seconds_per_bar)
        indices_split_bars = split_into_segments(n_frames, n_frames_per_var)

        # Generate Pseudo Lyrics Data
        frame2lyrics = defaultdict(lambda: {"word":"[PAD]", "sentence":"[PAD]"})
        lyrics_vectors = []
        for segment, pseudo_sentence in zip(segments, pseudo_sentences):
            input_ids = tokenizer.encode(" ".join(pseudo_sentence), add_special_tokens=True)
            input_ids_tensor = torch.tensor([input_ids]).to(device)
            N = input_ids_tensor.shape[1]
            attention_mask = torch.ones(N, dtype=torch.long).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids_tensor, attention_mask=attention_mask)
                last_hidden_states = outputs.last_hidden_state
            sentence_vector = last_hidden_states[0, :].mean(dim=0).cpu().numpy()

            phrase_boundaries = []
            start = 1  
            for phrase in pseudo_sentence:
                num_tokens = len(tokenizer.encode(phrase, add_special_tokens=False))
                end = start + num_tokens
                phrase_boundaries.append((start, end))
                start = end
            phrase_vectors = []
            for start, end in phrase_boundaries:
                phrase_vector = last_hidden_states[0, start:end].mean(dim=0)
                phrase_vectors.append(phrase_vector.cpu().numpy())
            lyrics_vectors.append({"words":pseudo_sentence, "sentence_vector":sentence_vector[:], "word_vectors":phrase_vectors[:]})

            start_frame = segment[0]
            end_frame = segment[1]
            n_sentence_frames = end_frame - start_frame
            n_words = len(pseudo_sentence)
            sentence_segments = split_with_gaps_around(n_sentence_frames, n_words, min_segment_length=5, gap_range=(0, 10))
            for idx in range(start_frame, end_frame+1):
                frame2lyrics[idx]["sentence"] = " ".join(pseudo_sentence)
                frame2lyrics[idx]["sentence_vector"] = sentence_vector
            for _id, (sentence_segment, pseudo_word) in enumerate(zip(sentence_segments, pseudo_sentence)):
                start_word_frame = start_frame+sentence_segment[0]
                end_word_frame = start_frame+sentence_segment[1]
                for idx in range(start_word_frame, end_word_frame+1):
                    frame2lyrics[idx]["word"] = pseudo_word
                    frame2lyrics[idx]["word_vector"] = phrase_vectors[_id]

        # Write lyrics data 
        joblib.dump(lyrics_vectors, "./lyrics_data4clustering/%03d.joblib"%i, compress=3)
        song_items = []
        for start_bar, end_bar in indices_split_bars:
            bar_lyrics = [frame2lyrics[idx] for idx in range(start_bar, end_bar+1)]
            all_phrase_padded = all(d.get("word") == "[PAD]" for d in bar_lyrics)
            all_sentence_padded = all(d.get("sentence") == "[PAD]" for d in bar_lyrics)
            if all_phrase_padded == True and all_sentence_padded == True:
                continue
            bar_items = []
            for j, lyrics_dict in enumerate(bar_lyrics):
                idx = j+start_bar
                if lyrics_dict["word"] == "[PAD]":
                    lyrics_dict["word_vector"] = pad_token_vector
                if lyrics_dict["sentence"] == "[PAD]":
                    lyrics_dict["sentence_vector"] = pad_token_vector
                lyrics_dict["frame"] = idx
                bar_items.append(lyrics_dict)
            song_items.append(bar_items[:])
        joblib.dump(song_items, "./lyrics_data/%03d.joblib"%i, compress=3)
    sys.stderr.write("\n")
            



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-seed", "--seed", dest="seed", default=0, type=int, help="seed value")
    parser.add_argument("-n_vocabs", "--n_vocabs", dest="n_vocabs", default=100, type=int, help="vocabrary size")
    args = parser.parse_args()
    main(args)
