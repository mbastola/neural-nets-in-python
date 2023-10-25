import math
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
import warnings
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_utils import get_multi30k_data, Vocabulary, TranslationDataset, PadCollate, bleu

warnings.filterwarnings('ignore')

# For reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Preprocessing ---
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<s>'
EOS_TOKEN = '</s>'
UNK_TOKEN = '<unk>'

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')

def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

# --- Model Components ---

class ScaledDotProductAttention(nn.Module):
    def __init__(self, scale, dropout_rate=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1))
        scores = scores / self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_probs = F.softmax(scores, dim=-1)
        output = torch.matmul(self.dropout(attn_probs), value)
        return output, attn_probs

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "`d_model` must be divisible by `n_heads`"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = self.d_v = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(np.sqrt(self.d_k), dropout_rate)
    def split_heads(self, x):
        batch_size = x.size(0)
        return x.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
    def group_heads(self, x):
        batch_size = x.size(0)
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
    def forward(self, query, key, value, mask=None):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        if mask is not None:
            mask = mask.unsqueeze(1)
        x, attn = self.attention(Q, K, V, mask)
        x = self.group_heads(x)
        x = self.W_o(x)
        return x, attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_rate=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        x = self.dropout(F.relu(self.w_1(x)))
        x = self.w_2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout_rate=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # With batch_first=True, x is (batch_size, seq_len, d_model).
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, mask):
        x1, _ = self.attn_layer(x, x, x, mask)
        x = self.attn_layer_norm(x + self.dropout(x1))
        x1 = self.ff_layer(x)
        x = self.ff_layer_norm(x + self.dropout(x1))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, dropout_rate=0.1, max_len=5000):
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x, mask):
        x = self.tok_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.layer_norm(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.enc_attn_layer = MultiHeadAttention(d_model, n_heads, dropout_rate)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.ff_layer = PositionwiseFeedForward(d_model, d_ff, dropout_rate)
        self.ff_layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        x1, _ = self.attn_layer(x, x, x, tgt_mask)
        x = self.attn_layer_norm(x + self.dropout(x1))
        x1, attn = self.enc_attn_layer(x, memory, memory, src_mask)
        x = self.enc_attn_layer_norm(x + self.dropout(x1))
        x1 = self.ff_layer(x)
        x = self.ff_layer_norm(x + self.dropout(x1))
        return x, attn

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, pad_idx, dropout_rate=0.1, max_len=5000):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.tok_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_embedding = PositionalEncoding(d_model, dropout_rate, max_len)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout_rate)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    def forward(self, x, memory, src_mask, tgt_mask):
        x = self.tok_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_embedding(x)
        for layer in self.layers:
            x, attn = layer(x, memory, src_mask, tgt_mask)
        x = self.layer_norm(x)
        return x, attn

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, generator, pad_idx):
        super(Transformer, self).__init__()
        self.pad_idx = pad_idx
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
    
    def get_pad_mask(self, x, pad_idx):
        return (x != pad_idx).unsqueeze(-2)
    
    def get_subsequent_mask(self, x):
        seq_len = x.size(1)
        subsequent_mask = np.triu(np.ones((1, seq_len, seq_len)), k=1).astype(np.int8)
        return (torch.from_numpy(subsequent_mask) == 0).to(x.device)
    
    def forward(self, src, tgt):
        src_mask = self.get_pad_mask(src, self.pad_idx)
        tgt_mask = self.get_pad_mask(tgt, self.pad_idx) & self.get_subsequent_mask(tgt)
        enc_output = self.encoder(src, src_mask)
        dec_output, attn = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.generator(dec_output)
        return output, attn

class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.proj(x)
        return F.log_softmax(x, dim=-1)

# --- Helper Functions ---

def model_summary(model):
    print(f'# of trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    print(f'# of non-trainable params: {sum(p.numel() for p in model.parameters() if not p.requires_grad):,}')

def convert_ids_to_text(ids, vocab, eos_idx, unk_idx):
    if ids.dim() == 1:
        output_tokens = []
        for token_id_tensor in ids:
            token_id = token_id_tensor.item()
            if token_id == eos_idx:
                break
            else:
                output_tokens.append(vocab.itos[token_id])
        return output_tokens
    elif ids.dim() == 2:
        return [convert_ids_to_text(ids[i, :], vocab, eos_idx, unk_idx) for i in range(ids.size(0))]
    raise RuntimeError(f'ids has {ids.size()} dimensions, expected 2 dimensions')

def plot_metrics(train_perplexities, valid_perplexities, valid_bleu4_scores):
    """Plots training and validation metrics over epochs."""
    epochs = range(1, len(train_perplexities) + 1)
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_perplexities, 'b-o', label='Train Perplexity')
    plt.plot(epochs, valid_perplexities, 'r-o', label='Valid Perplexity')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, valid_bleu4_scores, 'g-o', label='Valid BLEU4')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU4 Score')
    plt.title('Validation BLEU4 Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def greedy_decode(model, sentence, max_len=100, en_text=None, de_text=None, SOS_IDX=None, EOS_IDX=None, PAD_IDX=None):
    model.eval()

    if isinstance(sentence, str):
        tokens = tokenize_en(sentence)
    else:
        tokens = [token.lower() for token in sentence]
    token_ids = [SOS_IDX] + en_text.numericalize(" ".join(tokens)) + [EOS_IDX]
    source = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    source_mask = model.get_pad_mask(source, PAD_IDX)
    with torch.no_grad():
        enc_output = model.encoder(source, source_mask)
    target_ids = [SOS_IDX]
    for i in range(max_len):
        target = torch.tensor(target_ids, dtype=torch.long).unsqueeze(0).to(device)
        target_mask = model.get_pad_mask(target, PAD_IDX) & model.get_subsequent_mask(target)
        with torch.no_grad():
            dec_output, attn = model.decoder(target, enc_output, source_mask, target_mask)
            output = model.generator(dec_output)
        target_id = output.argmax(dim=-1)[:, -1].item()
        target_ids.append(target_id)
        if target_id == EOS_IDX or len(target_ids) >= max_len:
            break
    target_tokens = [de_text.itos[id] for id in target_ids]
    attn = attn.squeeze(0).cpu().detach().numpy()
    return target_tokens[1:], attn

def plot_attention_scores(source, target, attention):
    n_heads = attention.shape[0]
    if isinstance(source, str):
        source = [token.lower() for token in source.split(" ")] + [EOS_TOKEN]
    else:
        source = [token.lower() for token in source] + [EOS_TOKEN]
    fig = plt.figure(figsize=(24, 12))
    for h, head in enumerate(attention):
        ax = fig.add_subplot(2, 4, h + 1)
        x = source
        y = target if h % 4 == 0 else []
        sns.heatmap(
            head, xticklabels=x, yticklabels=y, square=True,
            vmin=0.0, vmax=1.0, cbar=False, cmap="Blues", ax=ax,
        )
    plt.show()

# --- Optimizer ---

class NoamOptim(object):
    def __init__(self, optimizer, d_model, factor, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.factor = factor
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0
    def zero_grad(self):
        self.optimizer.zero_grad()
    def step(self):
        self.n_steps += 1
        lr = self.get_lr()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()
    def get_lr(self):
        return self.factor * (
                self.d_model ** (-0.5)
                * min(self.n_steps ** (-0.5), self.n_steps * self.n_warmup_steps ** (-1.5))
        )

# --- Training & Evaluation Functions ---

def train_fn(model, iterator, optimizer, criterion, clip=1.0):
    model.train()
    total_loss = 0
    steps = 0
    tk0 = tqdm(iterator, total=len(iterator), position=0, leave=True)

    for idx, batch in enumerate(tk0):
        source, _ = batch.src
        target, _ = batch.trg
        source, target = source.to(device), target.to(device)

        optimizer.zero_grad()
        output, _ = model(source, target[:, :-1])

        loss = criterion(
            output.view(-1, output.size(-1)),
            target[:, 1:].contiguous().view(-1)
        )
        total_loss += loss.item()
        steps += 1

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        tk0.set_postfix(loss=total_loss / steps)

    tk0.close()
    return np.exp(total_loss / len(iterator))

def eval_fn(model, iterator, criterion, de_text, EOS_IDX, UNK_IDX):
    model.eval()
    total_loss = 0.0
    steps = 0
    hypotheses = []
    references = []
    tk0 = tqdm(iterator, total=len(iterator), position=0, leave=True)

    with torch.no_grad():
        for idx, batch in enumerate(tk0):
            source, _ = batch.src
            target, _ = batch.trg
            source, target = source.to(device), target.to(device)

            output, _ = model(source, target[:, :-1])

            loss = criterion(
                output.view(-1, output.size(-1)),
                target[:, 1:].contiguous().view(-1)
            )
            total_loss += loss.item()
            steps += 1

            output_ids = output.argmax(dim=-1)
            target_ids = target[:, 1:]

            pred_tokens = convert_ids_to_text(output_ids, de_text, EOS_IDX, UNK_IDX)
            target_tokens = convert_ids_to_text(target_ids, de_text, EOS_IDX, UNK_IDX)

            hypotheses.extend(pred_tokens)
            references.extend([[token] for token in target_tokens])

            tk0.set_postfix(loss=total_loss / steps)

    tk0.close()
    perplexity = np.exp(total_loss / len(iterator))
    bleu4 = bleu(hypotheses, references)
    return perplexity, bleu4

# --- Main Execution ---

def main():
    print(f"Using device: {device}")

    # --- Data Loading and Vocab ---
    (train_src_path, train_trg_path), \
    (valid_src_path, valid_trg_path), \
    (test_src_path, test_trg_path) = get_multi30k_data()

    specials = ["<unk>", "<pad>", "<s>", "</s>"]
    en_text = Vocabulary(tokenize_en, freq_threshold=2, specials=specials)
    de_text = Vocabulary(tokenize_de, freq_threshold=2, specials=specials)

    with open(train_src_path, encoding='utf-8') as f:
        src_corpus = f.readlines()
    with open(train_trg_path, encoding='utf-8') as f:
        trg_corpus = f.readlines()

    en_text.build_vocabulary([s.strip() for s in src_corpus])
    de_text.build_vocabulary([s.strip() for s in trg_corpus])

    train_data = TranslationDataset(train_src_path, train_trg_path, en_text, de_text)
    valid_data = TranslationDataset(valid_src_path, valid_trg_path, en_text, de_text)
    test_data = TranslationDataset(test_src_path, test_trg_path, en_text, de_text)
    
    PAD_IDX = de_text.stoi[PAD_TOKEN]
    SOS_IDX = de_text.stoi[SOS_TOKEN]
    EOS_IDX = de_text.stoi[EOS_TOKEN]
    UNK_IDX = de_text.stoi[UNK_TOKEN]

    print(f'English vocabulary size: {len(en_text)} words')
    print(f'German vocabulary size: {len(de_text)} words')

    # --- DataLoaders ---
    BATCH_SIZE = 128
    collate_fn = PadCollate(PAD_IDX)
    train_iterator = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # --- Model Hyperparameters & Instantiation ---
    INPUT_SIZE = len(en_text)
    OUTPUT_SIZE = len(de_text)
    HIDDEN_SIZE = 512
    N_LAYERS = 6
    N_HEADS = 8
    FF_SIZE = 2048
    DROPOUT_RATE = 0.1
    N_EPOCHS = 10
    CLIP = 1.0

    encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, PAD_IDX, DROPOUT_RATE)
    decoder = Decoder(OUTPUT_SIZE, HIDDEN_SIZE, N_LAYERS, N_HEADS, FF_SIZE, PAD_IDX, DROPOUT_RATE)
    generator = Generator(HIDDEN_SIZE, OUTPUT_SIZE)
    model = Transformer(encoder, decoder, generator, PAD_IDX).to(device)
    
    print("\nModel Summary:")
    model_summary(model)

    # --- Optimizer and Loss ---
    optimizer = NoamOptim(
        optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9),
        model.encoder.d_model, 2, 4000
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # --- Training Loop ---
    best_bleu4 = float('-inf')
    es_patience = 3
    patience = 0
    model_path = 'model.pth'

    train_perplexities = []
    valid_perplexities = []
    valid_bleu4_scores = []

    for epoch in range(N_EPOCHS):
        print(f'\nEpoch: {epoch+1}/{N_EPOCHS}')
        train_perplexity = train_fn(model, train_iterator, optimizer, criterion, CLIP)
        valid_perplexity, valid_bleu4 = eval_fn(model, valid_iterator, criterion, de_text, EOS_IDX, UNK_IDX)

        train_perplexities.append(train_perplexity)
        valid_perplexities.append(valid_perplexity)
        valid_bleu4_scores.append(valid_bleu4)

        print(f'Train perplexity: {train_perplexity:.4f}, Valid perplexity: {valid_perplexity:.4f}, Valid BLEU4: {valid_bleu4:.4f}')
        is_best = valid_bleu4 > best_bleu4
        if is_best:
            print(f'BLEU score improved ({best_bleu4:.4f} -> {valid_bleu4:.4f}). Saving Model!')
            best_bleu4 = valid_bleu4
            patience = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience += 1
            print(f'Early stopping counter: {patience} out of {es_patience}')
            if patience == es_patience:
                print(f'Early stopping! Best BLEU4: {best_bleu4:.4f}')
                break

    # --- Final Evaluation and Visualization ---
    model.load_state_dict(torch.load(model_path, map_location=device))
    print('\nEvaluating the model on test data ...')
    test_perplexity, test_bleu4 = eval_fn(model, test_iterator, criterion, de_text, EOS_IDX, UNK_IDX)
    print(f'Test perplexity: {test_perplexity:.4f}, Test BLEU4: {test_bleu4:.4f}')
    
    plot_metrics(
        train_perplexities=train_perplexities,
        valid_perplexities=valid_perplexities,
        valid_bleu4_scores=valid_bleu4_scores
    )

    # --- Inference Examples ---
    print("\n--- Inference Examples ---")
    example_indices = [7, 24, 50]
    for example_idx in example_indices:
        source_sentence = test_data.src_sentences[example_idx].strip()
        target_sentence = test_data.trg_sentences[example_idx].strip()

        predicted_tokens, attention_scores = greedy_decode(model, source_sentence, en_text=en_text, de_text=de_text, SOS_IDX=SOS_IDX, EOS_IDX=EOS_IDX, PAD_IDX=PAD_IDX)

        print(f'\nSource: {source_sentence}')
        print(f'Target: {target_sentence}')
        print(f'Predicted: {" ".join(predicted_tokens).replace(" </s>", "")}')

        source_tokens = tokenize_en(source_sentence)
        plot_attention_scores(source_tokens, predicted_tokens, attention_scores)


if __name__ == '__main__':
    main()