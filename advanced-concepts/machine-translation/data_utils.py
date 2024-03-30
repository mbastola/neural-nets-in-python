
import os
from io import open
import torch
import requests
import zipfile
from collections import Counter
import nltk
from tqdm import tqdm
import math
from datasets import load_dataset

# Download NLTK's tokenizer models if not already present.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print('NLTK "punkt" model not found. Downloading...')
    nltk.download('punkt')

def download_and_extract_wikitext2(path):
    """
    Downloads and extracts the WikiText-2 dataset.
    """
    data_path = os.path.join(path, 'wikitext-2')
    if os.path.exists(data_path):
        print(f'Dataset already found at {data_path}')
        return data_path

    os.makedirs(path, exist_ok=True)
    url = 'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip'
    zip_path = os.path.join(path, 'wikitext-2-v1.zip')
    
    if not os.path.exists(zip_path):
        print(f'Downloading {url} to {zip_path}')
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            with open(zip_path, 'wb') as f, tqdm(
                total=total_size, unit='iB', unit_scale=True, desc="Downloading WikiText-2"
            ) as pbar:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        print('Download complete.')

    print(f'Extracting {zip_path}')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path)
    print('Extraction complete.')
    return data_path

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        # Add special tokens that are not in the training data
        self.add_word('<unk>')
        self.add_word('<pad>')

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class BPTTBatchIterator:
    def __init__(self, source, bptt_len):
        self.source = source
        self.bptt_len = bptt_len
        self.i = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.source.size(0) - 1:
            raise StopIteration
        
        seq_len = min(self.bptt_len, len(self.source) - 1 - self.i)
        data = self.source[self.i:self.i+seq_len]
        target = self.source[self.i+1:self.i+1+seq_len]
        
        self.i += seq_len
        
        return data, target

    def __len__(self):
        return math.ceil((self.source.size(0) - 1) / self.bptt_len)

class Corpus(object):
    def __init__(self, batch_size, bptt_len, device):
        self.device = device
        self.batch_size = batch_size
        self.bptt_len = bptt_len

        self.dictionary = Dictionary()
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

        self.train_data = dataset["train"]
        self.valid_data = dataset["validation"]
        self.test_data = dataset["test"]

        self.train_iter = BPTTBatchIterator(self.train_data, self.bptt_len)
        self.valid_iter = BPTTBatchIterator(self.valid_data, self.bptt_len)
        self.test_iter = BPTTBatchIterator(self.test_data, self.bptt_len)

        self.vocab_size = len(self.dictionary)
        self.padding_idx = self.dictionary.word2idx['<pad>']

    def _read_tokens(self, path):
        """Reads a tokenized text file and adds <eos> token at the end of each line."""
        assert os.path.exists(path)
        with open(path, 'r', encoding="utf8") as f:
            tokens = []
            for line in f:
                words = line.split() + ['<eos>']
                tokens.extend(words)
        return tokens

    def _build_vocab(self, tokens):
        counter = Counter(tokens)
        sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        for word, _ in sorted_counter:
            self.dictionary.add_word(word)

    def numericalize(self, tokens):
        unk_idx = self.dictionary.word2idx['<unk>']
        ids = [self.dictionary.word2idx.get(token, unk_idx) for token in tokens]
        return torch.tensor(ids).type(torch.int64)

    def batchify(self, data, bsz):
        nbatch = data.size(0) // bsz
        data = data.narrow(0, 0, nbatch * bsz)
        data = data.view(bsz, -1).t().contiguous()
        return data.to(self.device)
