import os
from tqdm import tqdm
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import corpus_bleu
from datasets import load_dataset

# --- Data Downloading and Extraction ---

def get_multi30k_data():
    """Load Multi30k dataset from Hugging Face and save to local files."""
    data_dir = ".data"
    os.makedirs(data_dir, exist_ok=True)

    # Define file paths the rest of the script expects
    train_en_path = os.path.join(data_dir, "train.en")
    train_de_path = os.path.join(data_dir, "train.de")
    valid_en_path = os.path.join(data_dir, "val.en")
    valid_de_path = os.path.join(data_dir, "val.de")
    test_en_path = os.path.join(data_dir, "test2016.en")
    test_de_path = os.path.join(data_dir, "test2016.de")

    # Check if all files already exist to avoid re-downloading
    all_files = [train_en_path, train_de_path, valid_en_path, valid_de_path, test_en_path, test_de_path]
    if all(os.path.exists(p) for p in all_files):
        print("Multi30k files already exist. Skipping download and processing.")
        return (train_en_path, train_de_path), (valid_en_path, valid_de_path), (test_en_path, test_de_path)

    print("Downloading Multi30k dataset from Hugging Face...")
    dataset = load_dataset("bentrevett/multi30k")
    print(dataset)
    # Save splits to files in the format expected by the rest of the script
    splits = {
        'train': (dataset['train'], train_en_path, train_de_path),
        'validation': (dataset['validation'], valid_en_path, valid_de_path),
        'test': (dataset['test'], test_en_path, test_de_path)
    }

    for split_name, (data, en_path, de_path) in splits.items():
        with open(en_path, 'w', encoding='utf-8') as f_en, open(de_path, 'w', encoding='utf-8') as f_de:
            for example in tqdm(data, desc=f"Writing {split_name} split"):
                f_en.write(example['en'] + '\n')
                f_de.write(example['de'] + '\n')

    print("Dataset processing complete.")
    return (train_en_path, train_de_path), (valid_en_path, valid_de_path), (test_en_path, test_de_path)


# --- Vocabulary Class (replaces Field) ---

class Vocabulary:
    def __init__(self, tokenizer, freq_threshold, specials):
        self.tokenizer = tokenizer
        self.freq_threshold = freq_threshold
        self.specials = specials
        self.itos = {i: s for i, s in enumerate(specials)}
        self.stoi = {s: i for i, s in enumerate(specials)}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        for sentence in sentence_list:
            for word in self.tokenizer(sentence):
                frequencies[word] += 1

        idx = len(self.specials)
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokenized_text]


# --- Dataset and DataLoader (replaces Multi30k.splits and BucketIterator) ---

class TranslationDataset(Dataset):
    def __init__(self, src_path, trg_path, src_vocab, trg_vocab):
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        with open(src_path, encoding='utf-8') as f:
            self.src_sentences = f.readlines()
        with open(trg_path, encoding='utf-8') as f:
            self.trg_sentences = f.readlines()

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, index):
        src_sentence = self.src_sentences[index].strip().lower()
        trg_sentence = self.trg_sentences[index].strip().lower()

        src_numericalized = [self.src_vocab.stoi["<s>"]] + self.src_vocab.numericalize(src_sentence) + [self.src_vocab.stoi["</s>"]]
        trg_numericalized = [self.trg_vocab.stoi["<s>"]] + self.trg_vocab.numericalize(trg_sentence) + [self.trg_vocab.stoi["</s>"]]

        return torch.tensor(src_numericalized), torch.tensor(trg_numericalized)

class PadCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Sort by source length
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sources, targets = zip(*batch)

        padded_sources = pad_sequence(sources, batch_first=True, padding_value=self.pad_idx)
        padded_targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        
        # The original code ignored lengths, so we will too for compatibility.
        # We'll return a simple object-like structure to mimic batch.src and batch.trg
        class Batch:
            def __init__(self, src, trg):
                self.src = (src, None) # (data, lengths)
                self.trg = (trg, None)

        return Batch(padded_sources, padded_targets)

# --- BLEU Score ---
def bleu(hypotheses, references):
    """
    Calculates corpus-level BLEU score.
    hypotheses: list of lists of strings (predicted sentences)
    references: list of lists of lists of strings (reference sentences)
    """
    return corpus_bleu(references, hypotheses)