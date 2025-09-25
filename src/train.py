from utils.text_processing import text_processing
import json
from collections import Counter
from tqdm import tqdm
import os
import gdown

file_path = 'data/test.ft.txt'
file_id = '1h-pfNVXM7wGimFaFMdhBVLRqqdhrVn17'
url = f'https://drive.google.com/uc?id={file_id}'

if not os.path.exists(file_path):
    print(f"{file_path} not found. Downloading from Google Drive...")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    gdown.download(url, file_path, quiet=False)
    print("Download complete.")
else:
    print(f"{file_path} already exists.")

label_map = {
    '__label__1': 'negative',
    '__label__2': 'positive'
}

print("Reading and processing file lines...")
texts, sentiments = [], []
with open('data/test.ft.txt', 'r', encoding='utf-8') as file:
    for line in tqdm(file, desc="Reading lines"):
        line = line.strip()
        if not line:
            continue
        label, text = line.split(' ', 1)
        sentiment = label_map.get(label)
        if sentiment:
            sentiments.append(sentiment)
            texts.append(text.strip())

print("Processing text data...")
X = [text_processing(text) for text in tqdm(texts, desc="Text processing")]
Y = sentiments

print("Calculating priors...")
counts = Counter(Y)
n = len(Y)
priors = {label: counts[label] / n for label in counts}
print(f"Priors: {priors}")

print("Building vocabulary and counting word occurrences...")
word_counts_table = {
    'positive': Counter(),
    'negative': Counter(),
    't_positive': 0,
    't_negative': 0
}

for sentence, label in tqdm(zip(X, Y), desc="Counting words", total=len(X)):
    word_counts_table[label].update(sentence)
    word_counts_table[f't_{label}'] += len(sentence)

# Extract vocabulary (union of both counters)
vocab = set(word_counts_table['positive']) | set(word_counts_table['negative'])
print(f"Vocabulary size: {len(vocab)}")

print("Saving model data to JSON...")
data_to_save = {
    "priors": priors,
    "vocab": list(vocab),
    "word_counts_table": {
        'positive': dict(word_counts_table['positive']),
        'negative': dict(word_counts_table['negative']),
        't_positive': word_counts_table['t_positive'],
        't_negative': word_counts_table['t_negative']
    }
}

with open('models/model_data.json', 'w') as f:
    json.dump(data_to_save, f, ensure_ascii=False, separators=(',', ':'))

print("Done.")
