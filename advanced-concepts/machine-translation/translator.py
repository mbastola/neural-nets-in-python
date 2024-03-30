import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset, DatasetDict
from collections import Counter, OrderedDict
import spacy
import numpy as np

import random
import math
import time

# Set a random seed for reproducibility.
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load the cleaned dataset
dataset = load_dataset("parquet", data_files="./nep_to_eng_clean.parquet")

# The dataset only has a 'train' split. We need to create validation and test splits.
# First, split 'train' into training (85%) and a validation set (25% of the training set).
train_test_split = dataset['train'].train_test_split(test_size=0.15, seed=SEED)
train_valid_split = train_test_split['train'].train_test_split(test_size=0.25, seed=SEED)

# Combine these into a single DatasetDict object.
dataset = DatasetDict({
    'train': train_valid_split['train'],
    'validation': train_valid_split['test'],
    'test': train_test_split['test']
})

# Extract sentences into lists
train_np, train_en = dataset['train']['np'], dataset['train']['en']
valid_np, valid_en = dataset['validation']['np'], dataset['validation']['en']
test_np, test_en = dataset['test']['np'], dataset['test']['en']

#Tokenization
spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_np(text):
    """
    Tokenizes Nepali text from a string into a list of strings (tokens).
    A simple whitespace tokenizer for demonstration.
    """
    return text.split(' ')

# --- Vocabulary Building ---
class Vocabulary:
    def __init__(self, specials=['<unk>', '<pad>', '<sos>', '<eos>']):
        self.word2idx = OrderedDict()
        self.idx2word = []
        self.specials = specials
        self.unk_token = '<unk>'
        
        for special in self.specials:
            self.add_word(special)
        
        self.unk_index = self.word2idx[self.unk_token]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

    def build_vocab_from_iterator(self, iterator, min_freq=1):
        counter = Counter()
        for tokens in iterator:
            counter.update(tokens)
        
        # Sort by frequency, then alphabetically for stability
        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        
        for word, freq in sorted_by_freq_tuples:
            if freq >= min_freq:
                self.add_word(word)
    
    def __len__(self):
        return len(self.idx2word)

    def __call__(self, tokens):
        return [self.word2idx.get(token, self.unk_index) for token in tokens]

    def __getitem__(self, token):
        return self.word2idx.get(token, self.unk_index)

    def lookup_tokens(self, indices):
        return [self.idx2word[index] for index in indices]

def yield_tokens(data_iter, tokenizer):
    for text in data_iter:
        yield tokenizer(text.lower())

# --- Custom Dataset and Collation ---
class TranslationDataset(Dataset):
    def __init__(self, src_data, trg_data, src_vocab, trg_vocab, src_tokenizer, trg_tokenizer):
        self.src_data = src_data
        self.trg_data = trg_data
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer
        self.trg_tokenizer = trg_tokenizer

    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        src_text = self.src_data[idx].lower()
        trg_text = self.trg_data[idx].lower()
        
        src_tokens = ['<sos>'] + self.src_tokenizer(src_text) + ['<eos>']
        trg_tokens = ['<sos>'] + self.trg_tokenizer(trg_text) + ['<eos>']
        
        src_indices = self.src_vocab(src_tokens)
        trg_indices = self.trg_vocab(trg_tokens)
        
        return torch.tensor(src_indices), torch.tensor(trg_indices)

def collate_fn(batch, src_pad_idx, trg_pad_idx):
    src_batch, trg_batch = zip(*batch)
    
    src_padded = pad_sequence(list(src_batch), padding_value=src_pad_idx, batch_first=True)
    trg_padded = pad_sequence(list(trg_batch), padding_value=trg_pad_idx, batch_first=True)
    return src_padded, trg_padded

def filter_long_sentences(src_sents, trg_sents, tokenizer_src, tokenizer_trg, max_len):
    filtered_src, filtered_trg = [], []
    for src, trg in zip(src_sents, trg_sents):
        if len(tokenizer_src(src)) < max_len and len(tokenizer_trg(trg)) < max_len:
            filtered_src.append(src)
            filtered_trg.append(trg)
    return filtered_src, filtered_trg

#Model Building

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=160):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is [batch_size, seq_len, d_model]
        x = x.permute(1, 0, 2) # [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x.permute(1, 0, 2) # [batch_size, seq_len, d_model]

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=160):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, dropout, max_length)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=n_heads, dim_feedforward=pf_dim, dropout=dropout, batch_first=True)
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        embedding = self.tok_embedding(src) * self.scale
        pos_encoded_embedding = self.pos_embedding(embedding)
        encoded_src = self.layers(pos_encoded_embedding, src_key_padding_mask=src_mask)
        return encoded_src

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=160):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, dropout, max_length)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hid_dim, nhead=n_heads, dim_feedforward=pf_dim, dropout=dropout, batch_first=True)
        self.layers = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_sub_mask, trg_pad_mask, src_mask):
        pos_encoded_embedding = self.pos_embedding(self.tok_embedding(trg) * self.scale)
        output = self.layers(pos_encoded_embedding, enc_src, 
                             tgt_mask=trg_sub_mask, 
                             tgt_key_padding_mask=trg_pad_mask,
                             memory_key_padding_mask=src_mask)
        prediction = self.fc_out(output)
        return prediction

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # In PyTorch's transformer, True values are positions that will be masked.
        src_mask = (src == self.src_pad_idx)
        return src_mask
    
    def make_trg_mask(self, trg):
        trg_pad_mask = (trg == self.trg_pad_idx)
        trg_len = trg.shape[1]
        # In PyTorch's `nn.Transformer`, the mask is float('-inf') for masked positions.
        trg_sub_mask = torch.triu(torch.ones((trg_len, trg_len), device = self.device) * float('-inf'), diagonal=1)
        return trg_sub_mask, trg_pad_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_sub_mask, trg_pad_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_sub_mask, trg_pad_mask, src_mask)
        return output

#Model Training

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def train(model, iterator, optimizer, criterion, clip, device):
    model.train()
    epoch_loss = 0
    for i, (src, trg) in enumerate(iterator):
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:,:-1])
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_for_loss = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg_for_loss)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_for_loss = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg_for_loss)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

def translate_sentence(sentence, src_vocab, trg_vocab, src_tokenizer, model, device, max_len = 50):
    model.eval()
    tokens = [token.lower() for token in src_tokenizer(sentence)]
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indexes = src_vocab(tokens)
    src_tensor = torch.tensor(src_indexes, dtype=torch.long).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [trg_vocab['<sos>']]

    for i in range(max_len):
        trg_tensor = torch.tensor(trg_indexes, dtype=torch.long).unsqueeze(0).to(device)
        trg_sub_mask, trg_pad_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, trg_sub_mask, trg_pad_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        trg_indexes.append(pred_token)

        if pred_token == trg_vocab['<eos>']:
            break
    
    trg_tokens = trg_vocab.lookup_tokens(trg_indexes)
    return trg_tokens[1:]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #Create vocab
    SRC_VOCAB = Vocabulary(specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    SRC_VOCAB.build_vocab_from_iterator(yield_tokens(train_np, tokenize_np), min_freq=2)
    TRG_VOCAB = Vocabulary(specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    TRG_VOCAB.build_vocab_from_iterator(yield_tokens(train_en, tokenize_en), min_freq=2)
    print(f"Source (np) vocabulary size: {len(SRC_VOCAB)}")
    print(f"Target (en) vocabulary size: {len(TRG_VOCAB)}")

    SRC_PAD_IDX = SRC_VOCAB['<pad>']
    TRG_PAD_IDX = TRG_VOCAB['<pad>']

    #DataLoaders
    BATCH_SIZE = 32
    MAX_LEN = 160
    MAX_TOKENS = MAX_LEN - 2

    train_np_filtered, train_en_filtered = filter_long_sentences(train_np, train_en, tokenize_np, tokenize_en, MAX_TOKENS)
    valid_np_filtered, valid_en_filtered = filter_long_sentences(valid_np, valid_en, tokenize_np, tokenize_en, MAX_TOKENS)
    test_np_filtered, test_en_filtered = filter_long_sentences(test_np, test_en, tokenize_np, tokenize_en, MAX_TOKENS)

    train_dataset = TranslationDataset(train_np_filtered, train_en_filtered, SRC_VOCAB, TRG_VOCAB, tokenize_np, tokenize_en)
    valid_dataset = TranslationDataset(valid_np_filtered, valid_en_filtered, SRC_VOCAB, TRG_VOCAB, tokenize_np, tokenize_en)
    test_dataset = TranslationDataset(test_np_filtered, test_en_filtered, SRC_VOCAB, TRG_VOCAB, tokenize_np, tokenize_en)

    train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, SRC_PAD_IDX, TRG_PAD_IDX))
    valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, SRC_PAD_IDX, TRG_PAD_IDX))
    test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, SRC_PAD_IDX, TRG_PAD_IDX))

    print(f"Number of training examples: {len(train_dataset)}")
    print(f"Number of validation examples: {len(valid_dataset)}")
    print(f"Number of testing examples: {len(test_dataset)}")

    #Hyperparameters and Instantiation
    INPUT_DIM = len(SRC_VOCAB)
    OUTPUT_DIM = len(TRG_VOCAB)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 256
    DEC_PF_DIM = 256
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM, HID_DIM, ENC_LAYERS, ENC_HEADS, ENC_PF_DIM, ENC_DROPOUT, device, max_length=MAX_LEN)
    dec = Decoder(OUTPUT_DIM, HID_DIM, DEC_LAYERS, DEC_HEADS, DEC_PF_DIM, DEC_DROPOUT, device, max_length=MAX_LEN)
    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.apply(initialize_weights)

    LEARNING_RATE = 0.0005
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    #Training
    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):
        start_time = time.time()
        
        train_loss = train(model, train_iterator, optimizer, criterion, CLIP, device)
        valid_loss = evaluate(model, valid_iterator, criterion, device)
        
        end_time = time.time()
        
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) - (epoch_mins * 60))
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'transformer-nepali-english.pt')
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    #Inference
    model.load_state_dict(torch.load('transformer-nepali-english.pt'))

    for example_idx in range(10):
        src_text = test_np[example_idx]
        trg_text = test_en[example_idx]

        print(f'src = {src_text}')
        print(f'trg = {trg_text}')

        translation = translate_sentence(src_text, SRC_VOCAB, TRG_VOCAB, tokenize_np, model, device)

        print(f'predicted trg = {" ".join(translation)}\n')

if __name__ == '__main__':
    main()
