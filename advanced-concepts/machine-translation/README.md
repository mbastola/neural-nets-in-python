# Nepali to English Translation with a Transformer

In my previous project, we built the transformer architecture utilizing multi-headed attention. In this project, we advance our learning of transformer by building a machine translation model from the ground up. Insipired from the tutorial we did last time on Genman-to-english translation, I've chosen to translate from **Nepali (नेपाली)** to **English**. Similar to the last project, this one is inspired from the tutorial [notebook](https://colab.research.google.com/github/jaygala24/pytorch-implementations/blob/master/Attention%20Is%20All%20You%20Need.ipynb) but we will be using torch's inbuilt encoder and decoder classes instead. Unlike, the previous project where we built the transformer from scratch, this project aims to learn the existing pytorch library for transformers.


```python
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
```

### Downloading & Cleaning Dataset

I have downloaded a Nepali-English translation dataset from this Hugging Face [repo](https://huggingface.co/datasets/ashokpoudel/nepali-english-translation-dataset/tree/main). The dataset contained some incorrect labels and translations, and therefore we build a cleanup pipeline and save the clean data locally for our translation task.


```python
#let's set a random seed for reproducibility. 
SEED = 100
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
```


```python
#raw_dataset = load_dataset("ashokpoudel/nepali-english-translation-dataset")
dataset = load_dataset("parquet", data_files="./nep_to_eng_clean.parquet")
```


```python
dataset
```




    DatasetDict({
        train: Dataset({
            features: ['en', 'np'],
            num_rows: 1778698
        })
    })




```python
import regex as re

def contains_non_english_characters(text):
    """
    Checks if a string contains characters that are not Latin letters, numbers, punctuation, or ASCII characters.
    """
    pattern = re.compile(r'[^\p{IsLatin}\d\p{P}\p{S}\s]', re.UNICODE)
    return bool(pattern.search(text))

def check_for_non_latin(text):
    pattern = re.compile(r'[^\p{IsLatin}\d\s]', re.UNICODE)
    return bool(pattern.search(text))

def contains_non_english_characters_with_custom_symbols(text, custom_allowed_symbols=None):
    """
    Checks if a string contains characters that are not Latin letters,
    numbers, whitespace, or a user-defined list of common symbols.
    """
    if custom_allowed_symbols is None:
        custom_allowed_symbols = "`~!@#$%^&*()_+-=[]{}|\\;:'\",./<>?"

    pattern = re.compile(r'[^\p{IsLatin}\d\s' + re.escape(custom_allowed_symbols) + r']', re.UNICODE)
    return bool(pattern.search(text))

def remove_symbols_regex(text):
    """
    Removes all characters that are not letters, numbers, or spaces.
    """
    # [^a-zA-Z0-9\s] matches any character that is NOT a letter (a-z, A-Z),
    # a digit (0-9), or a whitespace character (\s).
    # It replaces these matched characters with an empty string.
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return clean_text

def contains_devanagari(text):
    """
    Checks if a string contains any Devanagari characters.
    """
    # The pattern searches for at least one Devanagari character.
    pattern = re.compile(r'\p{Devanagari}', re.UNICODE)
    return bool(pattern.search(text))

def contains_any_subsentence(text, sub_sentences):
    """
    Checks if a text contains any of the sub-sentences from a given list.
    """
    # Escape special characters in each sub_sentence and then join them with '|'
    pattern = '|'.join(re.escape(s) for s in sub_sentences)
    
    # Use re.search to find if the pattern exists anywhere in the text
    if re.search(pattern, text):
        return True
    else:
        return False

def check(text):
    #text2 = remove_symbols_regex(text)
    list_ch = ['Please translate "', 'here is the translated text into English']
    return contains_devanagari(text) or contains_any_subsentence(text, list_ch)
```


```python
# The dataset only has a 'train' split. We need to create validation and test splits.
# First, split 'train' into training (85%) and a vaildation set (.25 %) of the training set.
train_test_split = dataset['train'].train_test_split(test_size=0.15, seed=SEED)
train_valid_split = train_test_split['train'].train_test_split(test_size=0.25, seed=SEED)
```


```python
train_valid_split['train'], train_valid_split['test'], train_test_split['test']
```




    (Dataset({
         features: ['en', 'np'],
         num_rows: 1133919
     }),
     Dataset({
         features: ['en', 'np'],
         num_rows: 377974
     }),
     Dataset({
         features: ['en', 'np'],
         num_rows: 266805
     }))




```python
# Combine these into a single DatasetDict object.
dataset = DatasetDict({
    'train': train_valid_split['train'],
    'validation': train_valid_split['test'],
    'test': train_test_split['test']
})

# Let's inspect the new structure
print(dataset)
print("\nExample from training set:")
print(dataset['train'][0])

# Extract sentences into lists for our existing pipeline
train_np, train_en = dataset['train']['np'], dataset['train']['en']
valid_np, valid_en = dataset['validation']['np'], dataset['validation']['en'] 
test_np, test_en = dataset['test']['np'], dataset['test']['en']

```

    DatasetDict({
        train: Dataset({
            features: ['en', 'np'],
            num_rows: 1133919
        })
        validation: Dataset({
            features: ['en', 'np'],
            num_rows: 377974
        })
        test: Dataset({
            features: ['en', 'np'],
            num_rows: 266805
        })
    })
    
    Example from training set:
    {'en': 'Yas story tells that Naruto is a young ninja', 'np': 'यास कथा ले यो भन छ कि नारुतो एउता जवान निन्जा हुन् छ'}



```python
print([ (train_np[i],  train_en[i]) for i in range(5)])
```

    [('यास कथा ले यो भन छ कि नारुतो एउता जवान निन्जा हुन् छ', 'Yas story tells that Naruto is a young ninja'), ('मन्दिरमा एउटा आवाजले मलाई भन्यो, “हे मानिसको छोरो! यो ठाउँ मेरो सिंहासनको ठाउँ मेरो पैतालाको लागि हो, म यस ठाँउमा इस्राएलका मानिसहरूसित सदा-सर्वदा रहनेछु। इस्राएलका सन्तानले मेरो नाम फेरि नष्ट गर्ने छैनन्। राजाहरू अनि तिनीहरूका मानिसहरूले व्यभिचार पापहरू गरेर अथवा तिनीहरूको राजाहरूको मृत शरीरहरू यस ठाउँमा गाडेर मेरो नामलाई बदनाम गर्नेछैन।', 'He said to me, Son of man, [this is] the place of my throne, and the place of the soles of my feet, where I will dwell in the midst of the children of Israel forever. The house of Israel shall no more defile my holy name, neither they, nor their kings, by their prostitution, and by the dead bodies of their kings [in] their high places;'), ('यसै संगठनको माध्यमबाट चेतनाको दियो बाल्दैछन्।', 'Through this organization, consciousness is burning.'), ('१०देखि तीन ठूला निकाय-मानक एन्ड पुअरको, मुडीका लगानीकर्ता सेवा र स्विच मूल्याङ्कन सेवा।', "10 to the three largest agencies - Standard & Poor's, Moody's Investors Service and Fitch Ratings Service."), ('अन्तराष्ट्रिय समुदायको विश्वास आर्जन गर्न दलहरु मिल्नुपर्ने हो', 'The parties have to join to earn the trust of the international community')]


### Tokenization

The attention models do not see words; it sees numbers (tokens). The first step is to break down a sentence into a list of tokens. As done in previous project, we'll use `spacy` for English. For Nepali, a robust `spacy` model isn't available, so we'll start with a simple whitespace tokenizer. We could use a library like `sentencepiece` or build a custom tokenizer but that is out of scope for this project.


```python
spacy_en = spacy.load('en_core_web_sm')

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

def tokenize_np(text):
    """
    Tokenizes Nepali text from a string into a list of strings (tokens)
    A simple whitespace tokenizer for demonstration.
    """
    return text.split(' ')
```

### Vocabulary Building

The tutorial notebook used `torchtext` for vocabulary creation. Since its deprecated, we implement our own `Vocabulary` class similar to the last project. This class will handle mapping tokens to numerical indices and vice-versa. It will be responsible for managing special tokens like `<unk>` (for unknown words), `<pad>` (for padding), `<sos>` (start of sentence), and `<eos>` (end of sentence).


```python
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

# Create vocabularies for source and target languages
SRC_VOCAB = Vocabulary(specials=['<unk>', '<pad>', '<sos>', '<eos>'])
SRC_VOCAB.build_vocab_from_iterator(yield_tokens(train_np, tokenize_np), min_freq=2) # Use min_freq=2 for real data

TRG_VOCAB = Vocabulary(specials=['<unk>', '<pad>', '<sos>', '<eos>'])
TRG_VOCAB.build_vocab_from_iterator(yield_tokens(train_en, tokenize_en), min_freq=2)

print(f"Source (np) vocabulary size: {len(SRC_VOCAB)}")
print(f"Target (en) vocabulary size: {len(TRG_VOCAB)}")
```

    Source (np) vocabulary size: 357567
    Target (en) vocabulary size: 116944


### Custom Dataset and Collation

Next, we create a custom `torch.utils.data.Dataset`. This class will take our raw text data and convert it into tokenized and numericalized tensors on the fly. 


```python
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

SRC_PAD_IDX = SRC_VOCAB['<pad>']
TRG_PAD_IDX = TRG_VOCAB['<pad>']

def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    
    src_padded = pad_sequence(list(src_batch), padding_value=SRC_PAD_IDX, batch_first=True)
    trg_padded = pad_sequence(list(trg_batch), padding_value=TRG_PAD_IDX, batch_first=True)
    return src_padded, trg_padded
```


```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32 # Increased batch size for the larger dataset
MAX_LEN = 160 

# --- Filter out sentences longer than MAX_LEN ---
# crucial step to prevent errors with positional encoding and to manage memory.
# We reserve 2 tokens for <sos> and <eos>.
MAX_TOKENS = MAX_LEN - 2

def filter_long_sentences(src_sents, trg_sents, tokenizer_src, tokenizer_trg, max_len):
    filtered_src, filtered_trg = [], []
    for src, trg in zip(src_sents, trg_sents):
        if len(tokenizer_src(src)) < max_len and len(tokenizer_trg(trg)) < max_len:
            filtered_src.append(src)
            filtered_trg.append(trg)
    return filtered_src, filtered_trg

print(f"Original train examples: {len(train_np)}")
train_np, train_en = filter_long_sentences(train_np, train_en, tokenize_np, tokenize_en, MAX_TOKENS)
print(f"Filtered train examples: {len(train_np)}")

print(f"\nOriginal valid examples: {len(valid_np)}")
valid_np, valid_en = filter_long_sentences(valid_np, valid_en, tokenize_np, tokenize_en, MAX_TOKENS)
print(f"Filtered valid examples: {len(valid_np)}")

print(f"\nOriginal test examples: {len(test_np)}")
test_np, test_en = filter_long_sentences(test_np, test_en, tokenize_np, tokenize_en, MAX_TOKENS)
print(f"Filtered test examples: {len(test_np)}")

# Create dataset instances for each split
train_dataset = TranslationDataset(train_np, train_en, SRC_VOCAB, TRG_VOCAB, tokenize_np, tokenize_en)
valid_dataset = TranslationDataset(valid_np, valid_en, SRC_VOCAB, TRG_VOCAB, tokenize_np, tokenize_en)
test_dataset = TranslationDataset(test_np, test_en, SRC_VOCAB, TRG_VOCAB, tokenize_np, tokenize_en)

# Create DataLoader instances
train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
valid_iterator = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

print(f"Number of training examples: {len(train_dataset)}")
print(f"Number of validation examples: {len(valid_dataset)}")
print(f"Number of testing examples: {len(test_dataset)}")
```

    Original train examples: 1133919
    Filtered train examples: 1133792
    
    Original valid examples: 377974
    Filtered valid examples: 377911
    
    Original test examples: 266805
    Filtered test examples: 266766
    Number of training examples: 1133792
    Number of validation examples: 377911
    Number of testing examples: 266766


## 3. Building the Model

Finally, we have reached the exciting part! We will build the model piece by piece, following the architecture from the "Attention Is All You Need" paper but mostly utilizing the pytorch library.

### Positional Encoding


```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=MAX_LEN):
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
        # x is expected to be of shape [seq_len, batch_size, d_model]
        # but we use batch_first=True, so it's [batch_size, seq_len, d_model]

        x = x.permute(1, 0, 2) # [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x.permute(1, 0, 2) # [batch_size, seq_len, d_model]
```

### The Evaluation Loop


```python
def evaluate(model, iterator, criterion):
    
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
```

### Multi-Head Attention

Attention allows the model to look at different parts of the input sequence when producing an output. Multi-Head Attention runs the attention mechanism in parallel multiple times (heads). This allows the model to jointly attend to information from different representation subspaces at different positions. The outputs of each head are then concatenated and linearly transformed.

Fortunately, PyTorch's `nn.MultiheadAttention` layer implements this for us. We will use it within the `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer`.

### The Encoder

The encoder maps an input sequence of symbol representations (x₁, ..., xₙ) to a sequence of continuous representations z = (z₁, ..., zₙ). It's composed of a stack of identical layers. Each layer has two sub-layers: a multi-head self-attention mechanism, and a simple, position-wise fully connected feed-forward network.

Our `Encoder` consists of:
1.  An `Embedding` layer to turn input token indices into dense vectors.
2.  The `PositionalEncoding` we just defined.
3.  A stack of `nn.TransformerEncoderLayer`s, which contains the multi-head attention and feed-forward network.


```python
class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = MAX_LEN):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = PositionalEncoding(hid_dim, dropout, max_length)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=hid_dim, nhead=n_heads, dim_feedforward=pf_dim, dropout=dropout, batch_first=True)
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Scale the embedding and add positional encoding
        embedding = self.tok_embedding(src) * self.scale
        pos_encoded_embedding = self.pos_embedding(embedding)
        
        encoded_src = self.layers(pos_encoded_embedding, src_key_padding_mask=src_mask)
        
        #encoded_src = [batch size, src len, hid dim]
        
        return encoded_src
```

### The Decoder

The decoder is also composed of a stack of identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. This is how the decoder "looks at" the source sentence to generate the translation.

Crucially, the self-attention sub-layer in the decoder is modified to prevent positions from attending to subsequent positions. This masking, combined with the fact that the output embeddings are offset by one position, ensures that the predictions for position `i` can depend only on the known outputs at positions less than `i`.

Our `Decoder` consists of:
1.  An `Embedding` layer.
2.  `PositionalEncoding`.
3.  A stack of `nn.TransformerDecoderLayer`s, which contain self-attention, encoder-attention, and the feed-forward network.


```python
class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 device,
                 max_length = MAX_LEN):
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
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_sub_mask = [trg len, trg len]
        #trg_pad_mask = [batch size, trg len]
        #src_mask = [batch size, src len]
        
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        # Scale the embedding and add positional encoding
        pos_encoded_embedding = self.pos_embedding(self.tok_embedding(trg) * self.scale)
        
        output = self.layers(pos_encoded_embedding, enc_src, 
                             tgt_mask=trg_sub_mask, 
                             tgt_key_padding_mask=trg_pad_mask,
                             memory_key_padding_mask=src_mask)
        
        #output = [batch size, trg len, hid dim]
        
        prediction = self.fc_out(output)
        
        #prediction = [batch size, trg len, output dim]
        
        return prediction
```

### The Seq2Seq (Transformer) Model

Now we put it all together. Our `Seq2Seq` model will contain the encoder and decoder. It will also be responsible for creating the masks that are essential for the Transformer's operation.

-   **Source Mask**: This is a padding mask to ensure the model doesn't pay attention to `<pad>` tokens in the source sentence.
-   **Target Mask**: This is a combination of a padding mask and a subsequent mask. The subsequent mask prevents the decoder from "cheating" by looking at future tokens in the target sentence during training.


```python
class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        # src = [batch size, src len]
        # In PyTorch's transformer, True values are positions that will be masked.
        src_mask = (src == self.src_pad_idx)
        # src_mask = [batch size, src len]
        return src_mask
    
    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]
        trg_pad_mask = (trg == self.trg_pad_idx)
        # trg_pad_mask = [batch size, trg len]
        
        trg_len = trg.shape[1]
        
        # In PyTorch's `nn.Transformer`, the mask isfloat('-inf') for masked positions.
        trg_sub_mask = torch.triu(torch.ones((trg_len, trg_len), device = self.device) * float('-inf'), diagonal=1)
        
        # trg_sub_mask = [trg len, trg len]
        return trg_sub_mask, trg_pad_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_sub_mask, trg_pad_mask = self.make_trg_mask(trg)
        
        # The decoder in PyTorch needs both a `tgt_mask` (for subsequent positions) and a `tgt_key_padding_mask` (for padding).
        # The encoder needs a `src_key_padding_mask`.
        
        enc_src = self.encoder(src, src_mask)

        # Let's adjust the Decoder class to handle masks properly.
        output = self.decoder(trg, enc_src, trg_sub_mask, trg_pad_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        
        return output
```

## 4. Training the Model

Now we define our hyperparameters, instantiate the model, and set up the optimizer and loss function.

### Hyperparameters
These are chosen to be small for this demonstration. A real model would have a larger `HID_DIM`, `PF_DIM`, and more layers.


```python
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

# SRC_PAD_IDX and TRG_PAD_IDX are already defined

enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT, 
              device)

dec = Decoder(OUTPUT_DIM, 
              HID_DIM, 
              DEC_LAYERS, 
              DEC_HEADS, 
              DEC_PF_DIM, 
              DEC_DROPOUT, 
              device)

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
```

It's good practice to initialize the weights of the model. A well-chosen initialization can help with convergence.


```python
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

model.apply(initialize_weights)
```




    Seq2Seq(
      (encoder): Encoder(
        (tok_embedding): Embedding(357567, 256)
        (pos_embedding): PositionalEncoding(
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (layers): TransformerEncoder(
          (layers): ModuleList(
            (0-2): 3 x TransformerEncoderLayer(
              (self_attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
              )
              (linear1): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
              (linear2): Linear(in_features=256, out_features=256, bias=True)
              (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (dropout1): Dropout(p=0.1, inplace=False)
              (dropout2): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (dropout): Dropout(p=0.1, inplace=False)
      )
      (decoder): Decoder(
        (tok_embedding): Embedding(116944, 256)
        (pos_embedding): PositionalEncoding(
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (layers): TransformerDecoder(
          (layers): ModuleList(
            (0-2): 3 x TransformerDecoderLayer(
              (self_attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
              )
              (multihead_attn): MultiheadAttention(
                (out_proj): NonDynamicallyQuantizableLinear(in_features=256, out_features=256, bias=True)
              )
              (linear1): Linear(in_features=256, out_features=256, bias=True)
              (dropout): Dropout(p=0.1, inplace=False)
              (linear2): Linear(in_features=256, out_features=256, bias=True)
              (norm1): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (norm2): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (norm3): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
              (dropout1): Dropout(p=0.1, inplace=False)
              (dropout2): Dropout(p=0.1, inplace=False)
              (dropout3): Dropout(p=0.1, inplace=False)
            )
          )
        )
        (fc_out): Linear(in_features=256, out_features=116944, bias=True)
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )



### Optimizer and Loss Function

We'll use the Adam optimizer, which is a standard choice for Transformer models. For the loss function, `CrossEntropyLoss` is appropriate for this multi-class classification problem (predicting the next word). We must make sure to ignore the `<pad>` token in the target sequence when calculating the loss.


```python
LEARNING_RATE = 0.0005

optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
```

### The Training Loop

Here's where the magic happens. For each epoch, we iterate through our training data. 

The steps are:
1.  Get a batch of source and target sentences.
2.  Zero the gradients.
3.  Feed the source and target (with the last token `<eos>` removed) into the model to get a prediction.
4.  Calculate the loss between the model's prediction and the actual target (with the first token `<sos>` removed).
5.  Backpropagate the loss.
6.  Clip the gradients to prevent them from exploding.
7.  Update the model's weights.


```python
def train(model, iterator, optimizer, criterion, clip):
    
    model.train()
    
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):
        
        src = src.to(device)
        trg = trg.to(device)
        
        optimizer.zero_grad()
        
        # The target sequence for the decoder input should not have the <eos> token
        # The target sequence for the loss calculation should not have the <sos> token
        output = model(src, trg[:,:-1])
                
        #output = [batch size, trg len - 1, output dim]
        #trg = [batch size, trg len]
            
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg_for_loss = trg[:,1:].contiguous().view(-1)
                
        #output = [batch size * trg len - 1, output dim]
        #trg = [batch size * trg len - 1]
            
        loss = criterion(output, trg_for_loss)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)
```


```python
N_EPOCHS = 10 # Reduced for demonstration on the larger dataset
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    
    start_time = time.time()
    
    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    
    end_time = time.time()
    
    epoch_mins = int((end_time - start_time) / 60)
    epoch_secs = int((end_time - start_time) - (epoch_mins * 60))
    
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'transformer-nepali-english.pt')
    
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
```

    /home/mbastola/.pyenv/versions/lora/lib/python3.10/site-packages/torch/nn/functional.py:6041: UserWarning: Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.
      warnings.warn(
    /home/mbastola/.pyenv/versions/lora/lib/python3.10/site-packages/torch/nn/modules/transformer.py:515: UserWarning: The PyTorch API of nested tensors is in prototype stage and will change in the near future. We recommend specifying layout=torch.jagged when constructing a nested tensor, as this layout receives active development, has better operator coverage, and works with torch.compile. (Triggered internally at /pytorch/aten/src/ATen/NestedTensorImpl.cpp:178.)
      output = torch._nested_tensor_from_mask(


    Epoch: 01 | Time: 51m 59s
    	Train Loss: 3.837 | Train PPL:  46.371
    	 Val. Loss: 2.508 |  Val. PPL:  12.275
    Epoch: 02 | Time: 52m 10s
    	Train Loss: 2.567 | Train PPL:  13.025
    	 Val. Loss: 2.096 |  Val. PPL:   8.132
    Epoch: 03 | Time: 52m 29s
    	Train Loss: 2.269 | Train PPL:   9.673
    	 Val. Loss: 1.974 |  Val. PPL:   7.197
    Epoch: 04 | Time: 54m 25s
    	Train Loss: 2.130 | Train PPL:   8.412
    	 Val. Loss: 1.906 |  Val. PPL:   6.724
    Epoch: 05 | Time: 54m 52s
    	Train Loss: 2.040 | Train PPL:   7.693
    	 Val. Loss: 1.875 |  Val. PPL:   6.522
    Epoch: 06 | Time: 54m 9s
    	Train Loss: 1.977 | Train PPL:   7.220
    	 Val. Loss: 1.849 |  Val. PPL:   6.353
    Epoch: 07 | Time: 51m 30s
    	Train Loss: 1.928 | Train PPL:   6.874
    	 Val. Loss: 1.815 |  Val. PPL:   6.139
    Epoch: 08 | Time: 51m 17s
    	Train Loss: 1.890 | Train PPL:   6.619
    	 Val. Loss: 1.803 |  Val. PPL:   6.068
    Epoch: 09 | Time: 51m 28s
    	Train Loss: 1.859 | Train PPL:   6.420
    	 Val. Loss: 1.794 |  Val. PPL:   6.012
    Epoch: 10 | Time: 50m 52s
    	Train Loss: 1.835 | Train PPL:   6.262
    	 Val. Loss: 1.783 |  Val. PPL:   5.948


## 5. Inference

Now that we have a trained model, let's see how it performs. The inference process is different from training. We can't see the whole target sentence at once. Instead, we generate the translation one token at a time, in a loop.

The process:
1.  Encode the source sentence.
2.  Start the decoder input with the `<sos>` token.
3.  In a loop, pass the encoded source and the current decoder input into the model.
4.  Get the prediction for the next token (the one with the highest probability).
5.  If the token is `<eos>`, stop.
6.  Otherwise, append this predicted token to the decoder input and continue the loop.

This is a greedy decoding approach as we saw in the last project. More advanced methods like beam search can produce better results but are more complex.


```python
def translate_sentence(sentence, src_vocab, trg_vocab, src_tokenizer, model, device, max_len = 50):

    model.eval()
        
    if isinstance(sentence, str):
        tokens = [token.lower() for token in src_tokenizer(sentence)]
    else:
        tokens = [token.lower() for token in sentence]

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
```

Let's try translating one of the sentences from our training data. Since the model has overfit, it should be able to do this reasonably well.


```python
# Load the best model
model.load_state_dict(torch.load('transformer-nepali-english.pt'))

for example_idx in range(10):
#example_idx = 10

    src_text = test_np[example_idx]
    trg_text = test_en[example_idx]

    print(f'src = {src_text}')
    print(f'trg = {trg_text}')

    translation = translate_sentence(src_text, SRC_VOCAB, TRG_VOCAB, tokenize_np, model, device)

    print(f'predicted trg = {translation}\n')
```

    src = पहिलो फिल्मबाटै व्यवसायिक सफलता हात पारेपछि उनले अर्को फिल्म बनाए, ‘लूट’
    trg = After achieving business success from the first film, he made another film, 'Loot'
    predicted trg = ['after', 'handing', 'over', 'first', '-', 'business', 'success', 'in', 'the', 'first', 'quarter', ',', 'he', 'made', 'another', 'film', ',', '<unk>', '.', '<eos>']
    
    src = एएफसी सोलिडारिटी कप अन्तर्गत नेपालले भोलि पूर्वी टिमोरबिरुद्धको खेलसँगै प्रतियोगिताको सुरुवात गर्दैछ।
    trg = Under the AFC Solidarity Cup, Nepal is starting the competition with a game against East Timor tomorrow.
    predicted trg = ['under', 'the', 'afc', 'solidarity', 'cup', ',', 'nepal', 'is', 'starting', 'the', 'competition', 'with', 'the', 'eastern', '<unk>', 'game', 'tomorrow', '.', '<eos>']
    
    src = पछिल्लो समय चामल र अल-मल्की यस्तो नजिकको निकटतामा भएको र अनुहार बोल्न नसक्दा तुरुन्तै स्पष्ट थिएन।
    trg = It was not immediately clear when the last time Rice and al-Maliki have been in such close proximity and not spoken face-to-face.


    /home/mbastola/.pyenv/versions/lora/lib/python3.10/site-packages/torch/nn/functional.py:6041: UserWarning: Support for mismatched key_padding_mask and attn_mask is deprecated. Use same type for both instead.
      warnings.warn(


    predicted trg = ['last', 'time', 'rice', 'and', 'al', '-', 'maliki', 'were', 'in', 'such', 'close', 'proximity', 'and', 'nothing', 'quite', 'clear', 'when', 'the', 'face', 'could', 'not', 'speak', '.', '<eos>']
    
    src = मनोचिकित्सकका अनुसार उपचारका लागि आउनेहरुले घरपरिवार, साथीभाइ, आफन्तसँग प्रत्यक्ष भेटघाट नगर्ने, रातदिन एप चलाएर बस्ने, खाना समयमा नखाने, बिहान ढिला उठ्ने गरेको र पढाइ बिगारेको पाइएको छ
    trg = According to psychiatrist, those who come for treatment have been found to be households, friends, relatives who do not have direct meetings, run the app overnight, not eating at food time, delay in the morning and deteriorating their studies.
    predicted trg = ['according', 'to', 'the', '<unk>', ',', 'those', 'who', 'come', 'for', 'treatment', 'do', 'not', 'meet', 'directly', 'with', 'household', ',', 'friends', ',', 'relatives', ',', 'overnight', ',', 'not', 'live', 'in', 'the', 'app', ',', 'food', 'time', ',', 'delay', 'and', 'reading', 'has', 'been', 'damaged', '.', '<eos>']
    
    src = ढ. ःयगलतबष्ल ँष्निजत सञ्चालन
    trg = Wildlife conservation
    predicted trg = ['power', '<unk>', 'operation', '<eos>']
    
    src = यो सबैकोलागि होइन।
    trg = It's not for everybody.
    predicted trg = ['this', 'is', 'not', 'for', 'everyone', '.', '<eos>']
    
    src = तर, विश्वकपसम्म उनी फिट नहुने पक्का भएकाले अस्ट्रेलियन बोर्ड दु: खी बनेको छ।
    trg = However, the World Cup, he is sure to be fit because the Australian Board sad: sad has become.
    predicted trg = ['however', ',', 'he', 'is', 'sure', 'to', 'not', 'be', 'fit', 'in', 'the', 'world', 'cup', ',', 'australian', 'board', 'suffering', ':', 'sad', 'has', 'become', '.', '<eos>']
    
    src = किनभने उनीहरूले भनिरहेका छन्, यदि मैले आफूलाई स्वस्थ राखें भने लामो दौडमा म कम पैसा खर्च गरिरहेको छु।
    trg = Because they're saying, if I keep myself healthy, in the long run, I'm gonna be spending less money.
    predicted trg = ['because', 'they', "'re", 'saying', ',', 'if', 'i', 'kept', 'myself', 'healthy', ',', 'i', "'m", 'spending', 'less', 'money', 'in', 'a', 'long', 'run', '.', '<eos>']
    
    src = अष्ट्रेलियाको ब्याटिङ विश्व कप विजेताहरूले ३०६-६ चुनौती पोस्ट गरे जस्तै हेर्न सक्छन्, म्याच हद्दीन (८७ बाहिर छैन), सिमन्ड (८७) र मट्टे हेकेन (७५) सबैले भारतीय आक्रमण तरवारमा राखे।
    trg = Australia's batting might was on view as the World Cup winners posted a challenging 306-6, with man-of-the-match Haddin (87 not out), Symonds (87) and Matthew Hayden (75) all putting the Indian attack to the sword.
    predicted trg = ['australia', "'s", 'batting', 'world', 'cup', 'winners', 'can', 'see', 'as', 'they', 'posed', 'a', 'challenge', 'to', 'the', '<unk>', ',', 'not', 'out', 'of', 'the', 'match', '(', 'http', ':', '/', '/', '/', '<unk>', ')', ',', 'matthew', 'hayden', '(', '<unk>', ')', 'and', 'matthew', 'hayden', '(', '<unk>', ')', 'all', 'kept', 'the', 'indian', 'attack', 'on', 'the', 'horse']
    
    src = एमाओवादीले गृहलगायत महत्वपूर्ण मन्त्रालय पाउने भएकाले सोहीअनुसार मन्त्री चयन गर्ने बताएको छ
    trg = The UCPN (Maoist) has said that it will select a minister accordingly as it will get important ministries including home
    predicted trg = ['the', 'ucpn', '(', 'maoist', ')', 'has', 'said', 'that', 'minister', 'will', 'select', 'accordingly', 'as', 'the', 'important', 'ministry', 'including', 'home', 'minister', 'is', 'received', '<eos>']
    


## 6. Conclusion and Next Steps

Building the Transformer from its core components: the positional encoding, the encoder, and decoder, has been a fantastic experience and given me an appreciation of the details of how it works. The role of masking, to me in particular, is now much clearer.

**Limitations & Future Work:**

1.  **Dataset Size**: The most obvious limitation is our not so large dataset size. The next step in this tangent would be to find a substantial Nepali-English parallel corpus to train a genuinely useful model.
2.  **Tokenizer**: The whitespace tokenizer for Nepali is quite basic. A subword tokenizer like Byte-Pair Encoding (BPE) or SentencePiece would be far more effective, especially for handling morphology and unknown words.
3.  **Evaluation**: I only performed a qualitative check. For a serious project, I would need to implement a proper evaluation loop using a test set and calculate metrics like BLEU score.
4.  **Hyperparameter Tuning**: The current hyperparameters were chosen arbitrarily. A systematic search (e.g., using Optuna or Ray Tune) would be necessary to find the optimal configuration.
5.  **Beam Search**: Implementing beam search for decoding instead of the current greedy approach would likely improve translation quality.


```python

```
