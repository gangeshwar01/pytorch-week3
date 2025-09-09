# File: code/transformer_translation.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.metrics import bleu_score

import spacy
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "../runs/mt"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Data Preparation ---
# Load tokenizers
try:
    spacy_de = spacy.load('de_core_news_sm')
    spacy_en = spacy.load('en_core_web_sm')
except IOError:
    print("Spacy models not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    print("python -m spacy download de_core_news_sm")
    exit()

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]

def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

# Special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Build vocabulary
def yield_tokens(data_iter, tokenizer, language_index):
    for data_sample in data_iter:
        yield tokenizer(data_sample[language_index])

print("Loading Multi30k dataset and building vocabularies...")
train_iter = Multi30k(split='train', language_pair=('de', 'en'))
src_tokenizer = get_tokenizer(tokenize_de)
tgt_tokenizer = get_tokenizer(tokenize_en)

# Source language (German) vocabulary
train_iter_copy = Multi30k(split='train', language_pair=('de', 'en'))
src_vocab = build_vocab_from_iterator(yield_tokens(train_iter_copy, src_tokenizer, 0),
                                      min_freq=2, specials=special_symbols, special_first=True)
src_vocab.set_default_index(UNK_IDX)

# Target language (English) vocabulary
train_iter_copy = Multi30k(split='train', language_pair=('de', 'en'))
tgt_vocab = build_vocab_from_iterator(yield_tokens(train_iter_copy, tgt_tokenizer, 1),
                                      min_freq=2, specials=special_symbols, special_first=True)
tgt_vocab.set_default_index(UNK_IDX)

# --- Transformer Model Implementation ---

class PositionalEncoding(nn.Module):
    """ Sinusoidal positional encoding. """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # Register as buffer to avoid being part of model parameters

    def forward(self, x):
        """ x shape: [seq_len, batch_size, d_model] """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention mechanism from scratch. """
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(DEVICE)

    def forward(self, query, key, value, mask=None):
        # query, key, value shape: [batch_size, seq_len, d_model]
        batch_size = query.shape[0]

        # 1. Project Q, K, V
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # 2. Reshape for multi-head attention: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 3. Calculate attention scores (Scaled Dot-Product Attention)
        # energy shape: [batch_size, num_heads, query_len, key_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # 4. Apply mask (if provided)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        # 5. Softmax to get attention weights
        attention = torch.softmax(energy, dim=-1)
        attention = self.dropout(attention)

        # 6. Apply attention to V and concatenate heads
        # x shape: [batch_size, num_heads, query_len, head_dim]
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous() # [batch_size, query_len, num_heads, head_dim]
        x = x.view(batch_size, -1, self.d_model) # [batch_size, query_len, d_model]

        # 7. Final output projection
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedforward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Self-attention + residual connection + layer norm
        # Note: Using pre-norm (LayerNorm before attention) often leads to more stable training,
        # but post-norm (as implemented here) follows the original paper's diagram.
        _src, _ = self.self_attn(src, src, src, src_mask)
        src = self.norm1(src + self.dropout(_src))

        # Feedforward + residual connection + layer norm
        _src = self.feed_forward(src)
        src = self.norm2(src + self.dropout(_src))
        return src

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        # 1. Masked Self-Attention on target sequence
        _tgt, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(_tgt))

        # 2. Cross-Attention (Query from decoder, Key/Value from encoder)
        _tgt, attention = self.cross_attn(tgt, enc_src, enc_src, src_mask)
        tgt = self.norm2(tgt + self.dropout(_tgt))

        # 3. Feedforward network
        _tgt = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(_tgt))
        return tgt, attention

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, dropout, max_len=100):
        super(Encoder, self).__init__()
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(DEVICE)

    def forward(self, src, src_mask):
        # src shape: [batch_size, src_len]
        # src_mask shape: [batch_size, 1, 1, src_len] or [batch_size, 1, src_len, src_len]
        batch_size, src_len = src.shape
        src = self.dropout((self.tok_embedding(src) * self.scale))
        src = self.pos_encoding(src.permute(1, 0, 2)).permute(1, 0, 2) # [batch_size, src_len, d_model]

        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, num_layers, num_heads, d_ff, dropout, max_len=100):
        super(Decoder, self).__init__()
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(DEVICE)

    def forward(self, tgt, enc_src, tgt_mask, src_mask):
        # tgt shape: [batch_size, tgt_len]
        batch_size, tgt_len = tgt.shape
        tgt = self.dropout((self.tok_embedding(tgt) * self.scale))
        tgt = self.pos_encoding(tgt.permute(1, 0, 2)).permute(1, 0, 2) # [batch_size, tgt_len, d_model]

        for layer in self.layers:
            tgt, attention = layer(tgt, enc_src, tgt_mask, src_mask)

        output = self.fc_out(tgt)
        return output, attention

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, tgt_pad_idx, device):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

    def create_src_mask(self, src):
        # src shape: [batch_size, src_len]
        # mask shape: [batch_size, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def create_tgt_mask(self, tgt):
        # tgt shape: [batch_size, tgt_len]
        # Causal mask (look-ahead mask)
        tgt_len = tgt.shape[1]
        tgt_pad_mask = (tgt != self.tgt_pad_idx).unsqueeze(1).unsqueeze(2) # [batch_size, 1, 1, tgt_len]
        tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=self.device)).bool() # [tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & tgt_sub_mask # Broadcasts to [batch_size, 1, tgt_len, tgt_len]
        return tgt_mask

    def forward(self, src, tgt):
        src_mask = self.create_src_mask(src)
        tgt_mask = self.create_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        return output, attention

# --- Data Collator and Preprocessing ---

def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))

src_transform = sequential_transforms(src_tokenizer, src_vocab, tensor_transform)
tgt_transform = sequential_transforms(tgt_tokenizer, tgt_vocab, tensor_transform)

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_transform(src_sample.rstrip("\n")))
        tgt_batch.append(tgt_transform(tgt_sample.rstrip("\n")))
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch

# --- Training and Evaluation ---

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, tgt in dataloader:
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)

        # Target input for decoder: <bos> token1 token2 ... tokenN
        tgt_input = tgt[:, :-1]
        # Target output for loss calculation: token1 token2 ... tokenN <eos>
        tgt_output = tgt[:, 1:]

        optimizer.zero_grad()
        output, _ = model(src, tgt_input)

        # Reshape for CrossEntropyLoss: output[batch_size * seq_len, vocab_size], target[batch_size * seq_len]
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        tgt_output = tgt_output.contiguous().view(-1)

        loss = criterion(output, tgt_output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1) # Gradient clipping
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            output, _ = model(src, tgt_input)
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            tgt_output = tgt_output.contiguous().view(-1)

            loss = criterion(output, tgt_output)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# Greedy decoding function for translation generation
def translate_sentence(sentence, model, src_tokenizer, src_vocab, tgt_vocab, device, max_len=50):
    model.eval()
    tokens = [tok.text for tok in src_tokenizer(sentence)]
    token_ids = [BOS_IDX] + [src_vocab[token] for token in tokens] + [EOS_IDX]
    src_tensor = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    src_mask = model.create_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    tgt_indices = [BOS_IDX]
    for i in range(max_len):
        tgt_tensor = torch.LongTensor(tgt_indices).unsqueeze(0).to(device)
        tgt_mask = model.create_tgt_mask(tgt_tensor)

        with torch.no_grad():
            output, attention = model.decoder(tgt_tensor, enc_src, tgt_mask, src_mask)

        pred_token_id = output.argmax(2)[:, -1].item()
        tgt_indices.append(pred_token_id)

        if pred_token_id == EOS_IDX:
            break

    tgt_tokens = [tgt_vocab.get_itos()[i] for i in tgt_indices]
    return tgt_tokens[1:-1], attention # Exclude <bos> and <eos>

# --- Visualization Functions ---
def plot_loss_curves(train_losses, valid_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss Curves')
    plt.savefig(os.path.join(OUTPUT_DIR, "curves_mt.png"))
    plt.close()

def plot_attention_heatmap(src_tokens, tgt_tokens, attention_matrix, layer, head):
    # attention_matrix shape: [batch_size, num_heads, tgt_len, src_len] -> select head and squeeze batch dim
    heatmap = attention_matrix[0, head, :, :].cpu().numpy()

    fig, ax = plt.subplots(figsize=(max(len(src_tokens)//2, 8), max(len(tgt_tokens)//2, 8)))
    cax = ax.matshow(heatmap, cmap='viridis')
    fig.colorbar(cax)

    ax.set_xticks(range(len(src_tokens)))
    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_xticklabels([''] + src_tokens, rotation=90) # Add empty string for potential <bos> offset if needed
    ax.set_yticklabels([''] + tgt_tokens)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))
    ax.yaxis.set_major_locator(plt.MultipleLocator(1))

    plt.title(f'Attention Heatmap (Layer {layer}, Head {head})')
    plt.xlabel('Source Sequence')
    plt.ylabel('Target Sequence')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"attention_layer{layer}_head{head}.png"))
    plt.close()

def visualize_masks(model, test_loader):
    src, tgt = next(iter(test_loader))
    src, tgt = src.to(DEVICE), tgt.to(DEVICE)

    # Causal mask for target (decoder self-attention)
    tgt_mask = model.create_tgt_mask(tgt)

    # Padding mask for source (encoder-decoder attention)
    src_padding_mask = model.create_src_mask(src).squeeze(1).squeeze(1) # simplify for plotting [batch, seq_len]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot source padding mask
    axes[0].imshow(src_padding_mask[0].cpu().numpy(), cmap='gray')
    axes[0].set_title(f"Source Padding Mask (Sample 0)\nShape: {src_padding_mask[0].shape}")
    axes[0].set_xlabel("Source Token Index")
    axes[0].set_ylabel("Batch dimension (sliced)")

    # Plot target causal mask
    axes[1].imshow(tgt_mask[0].squeeze(0).cpu().numpy(), cmap='gray') # [1, tgt_len, tgt_len] -> [tgt_len, tgt_len]
    axes[1].set_title(f"Target Causal Mask (Sample 0)\nShape: {tgt_mask[0].squeeze(0).shape}")
    axes[1].set_xlabel("Key Token Index")
    axes[1].set_ylabel("Query Token Index")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "masks_demo.png"))
    plt.close()

def generate_translation_report(model, test_iter, num_samples=10):
    print("Generating translation examples and calculating BLEU score...")
    targets = []
    predictions = []
    sources = []
    data_samples = list(test_iter)

    for i in range(num_samples):
        src_sentence, tgt_sentence = data_samples[i]
        predicted_tokens, _ = translate_sentence(src_sentence, model, src_tokenizer, src_vocab, tgt_vocab, DEVICE)
        predicted_sentence = " ".join(predicted_tokens)

        sources.append(src_sentence)
        targets.append(tgt_sentence)
        predictions.append(predicted_sentence)

    # Generate table visualization
    fig, ax = plt.subplots(figsize=(12, num_samples * 0.5))
    ax.axis('off')
    col_labels = ["Source (German)", "Target (English)", "Prediction"]
    cell_text = [[s, t, p] for s, t, p in zip(sources, targets, predictions)]
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='left', colWidths=[0.4, 0.3, 0.3])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.title("Translation Examples Comparison")
    plt.savefig(os.path.join(OUTPUT_DIR, "decodes_table.png"), bbox_inches='tight')
    plt.close()

    # Calculate Corpus BLEU score for full test set
    full_targets = []
    full_predictions = []
    for src, tgt in data_samples:
        pred_tokens, _ = translate_sentence(src, model, src_tokenizer, src_vocab, tgt_vocab, DEVICE)
        full_targets.append([tgt.split()]) # Target must be list of lists of tokens
        full_predictions.append(pred_tokens)

    # Suppress warnings for short sentences in BLEU calculation
    warnings.filterwarnings("ignore", category=UserWarning)
    bleu = bleu_score(full_predictions, full_targets)
    warnings.filterwarnings("default")

    # Generate BLEU score report image
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    text = f"Corpus BLEU Score: {bleu*100:.2f}"
    ax.text(0.5, 0.5, text, horizontalalignment='center', verticalalignment='center', fontsize=20)
    plt.title("BLEU Score Evaluation")
    plt.savefig(os.path.join(OUTPUT_DIR, "bleu_report.png"), bbox_inches='tight')
    plt.close()

    return bleu

# --- Main Execution ---
def main():
    # Model Hyperparameters (small model for quick training)
    INPUT_DIM = len(src_vocab)
    OUTPUT_DIM = len(tgt_vocab)
    D_MODEL = 256  # Embed dim (original paper uses 512)
    D_FF = 512     # Feedforward inner dim (original paper uses 2048)
    NUM_LAYERS = 3 # Number of encoder/decoder layers (original paper uses 6)
    NUM_HEADS = 8
    DROPOUT = 0.1
    LEARNING_RATE = 0.0005
    NUM_EPOCHS = 10 # Increase to 20-30 for better BLEU score
    BATCH_SIZE = 64

    # Initialize model
    print("Initializing Transformer model...")
    encoder = Encoder(INPUT_DIM, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT)
    decoder = Decoder(OUTPUT_DIM, D_MODEL, NUM_LAYERS, NUM_HEADS, D_FF, DROPOUT)
    model = Transformer(encoder, decoder, PAD_IDX, PAD_IDX, DEVICE).to(DEVICE)
    model.apply(initialize_weights)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    # Load data iterators
    train_data = Multi30k(split='train', language_pair=('de', 'en'))
    valid_data = Multi30k(split='valid', language_pair=('de', 'en'))
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    # Training loop
    train_losses, valid_losses = [], []
    best_valid_loss = float('inf')

    print(f"Starting training for {NUM_EPOCHS} epochs on {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        valid_loss = evaluate(model, valid_dataloader, criterion)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'transformer_model.pth'))

        print(f'Epoch: {epoch+1:02} | Train Loss: {train_loss:.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f}')

    # Load best model for evaluation
    model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, 'transformer_model.pth')))

    # Generate visualizations
    print("Generating visualizations...")
    plot_loss_curves(train_losses, valid_losses)

    # Visualize masks using test data loader
    test_data = Multi30k(split='test', language_pair=('de', 'en'))
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    visualize_masks(model, test_dataloader)

    # Generate translation examples and BLEU score report
    test_iter = Multi30k(split='test', language_pair=('de', 'en'))
    bleu_val = generate_translation_report(model, test_iter, num_samples=10)
    print(f"Final Corpus BLEU Score: {bleu_val*100:.2f}")

    # Generate attention heatmap for a sample sentence
    test_iter_list = list(Multi30k(split='test', language_pair=('de', 'en')))
    sample_src, sample_tgt = test_iter_list[0]
    predicted_tokens, attention = translate_sentence(sample_src, model, src_tokenizer, src_vocab, tgt_vocab, DEVICE)
    # Get attention from the last decoder layer (attention shape: [batch_size, num_heads, tgt_len, src_len])
    plot_attention_heatmap(tokenize_de(sample_src), predicted_tokens, attention, layer=NUM_LAYERS-1, head=0)

if __name__ == "__main__":
    main()
