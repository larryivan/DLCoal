import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import multiprocessing
import os
from datetime import datetime
import math

from data_simulator import DataSimulator

class PSMCDataset(Dataset):
    def __init__(self, sequences, tmrca_distributions):
        self.sequences = sequences
        self.tmrca_distributions = tmrca_distributions

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.tmrca_distributions[idx]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerPSMC(nn.Module):
    def __init__(self, output_size, chunk_size=2048, embedding_dim=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024, dropout_rate=0.2):
        super(TransformerPSMC, self).__init__()
        self.chunk_size = chunk_size
        self.embedding_dim = embedding_dim
        
        # 1. CNN Chunk Encoder
        self.chunk_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=16, padding='same'), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=16, padding='same'), nn.ReLU(), nn.BatchNorm1d(64), nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, embedding_dim)
        )
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout_rate)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward, dropout_rate, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 4. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embedding_dim // 2, output_size)
        )
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        batch_size = x.shape[0]
        
        # Ensure sequence length is divisible by chunk_size
        if x.shape[1] % self.chunk_size != 0:
            raise ValueError(f"Sequence length {x.shape[1]} is not divisible by chunk size {self.chunk_size}")

        # Chunk the input
        x = x.view(batch_size, -1, self.chunk_size) # -> (batch_size, num_chunks, chunk_size)
        num_chunks = x.shape[1]
        
        # Encode chunks with CNN
        x = x.view(-1, 1, self.chunk_size) # -> (batch*num_chunks, 1, chunk_size)
        chunk_embeddings = self.chunk_encoder(x) # -> (batch*num_chunks, embedding_dim)
        chunk_embeddings = chunk_embeddings.view(batch_size, num_chunks, self.embedding_dim) # -> (batch, num_chunks, embedding_dim)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, chunk_embeddings), dim=1) # -> (batch, num_chunks+1, embedding_dim)
        
        # Add positional encoding
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1) # Transformer expects (seq_len, batch, dim) for pos encoding, but our layer is batch_first
        
        # Transformer encoding
        transformer_output = self.transformer_encoder(x) # -> (batch, num_chunks+1, embedding_dim)
        
        # Get CLS token output for classification
        cls_output = transformer_output[:, 0, :] # -> (batch, embedding_dim)
        
        # Final classification
        logits = self.classifier(cls_output)
        return self.log_softmax(logits)

def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.long())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

def plot_psmc_comparison(model, test_samples, num_plots=5):
    model.eval()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = f"plots_transformer_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plots will be saved to '{plot_dir}/'")

    for i in range(min(num_plots, len(test_samples))):
        sample = test_samples[i]
        input_seq = torch.from_numpy(sample['binary_seq']).unsqueeze(0)
        
        with torch.no_grad():
            predicted_log_dist = model(input_seq.long())
        
        predicted_dist = torch.exp(predicted_log_dist).numpy().flatten()
        fig, ax = plt.subplots(figsize=(12, 7))

        # Plotting logic is the same as before...
        true_times = [0]; true_ne = [sample['Ne']]
        for event in sorted(sample['demographic_events'], key=lambda e: e.time):
            true_times.append(event.time); true_ne.append(event.initial_size)
        ax.step(true_times, true_ne, where='post', label='True Ne History', color='green', linewidth=2.5, zorder=2)
        
        bin_edges = sample['bin_edges']; bin_widths = np.diff(bin_edges)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        predicted_dist_nonzero = predicted_dist + 1e-12
        relative_ne_pred = bin_widths / predicted_dist_nonzero
        
        visible_true_ne = np.array(true_ne)[np.array(true_times) < bin_edges[-1]]
        mean_true_ne = np.mean(visible_true_ne) if len(visible_true_ne) > 0 else sample['Ne']
        scaling_factor = mean_true_ne / np.mean(relative_ne_pred)
        ne_pred = relative_ne_pred * scaling_factor

        ax.step(bin_midpoints, ne_pred, where='mid', label='Predicted Ne History', color='purple', linestyle='--', linewidth=2, zorder=3)
        
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('Time (generations ago) [Log Scale]'); ax.set_ylabel('Effective Population Size (Ne) [Log Scale]')
        ax.set_title(f"PSMC Comparison for Sample {i+1} (Model: {sample['model_type']})"); ax.legend()
        ax.grid(True, which="both", ls="--", c='0.7', zorder=0)
        ax.set_xlim(left=max(10, bin_edges[1] * 0.5), right=bin_edges[-1] * 1.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"transformer_comparison_sample_{i+1}.png"), dpi=300)
        plt.close(fig)

if __name__ == '__main__':
    try:
        num_threads = multiprocessing.cpu_count()
        torch.set_num_threads(num_threads)
        print(f"PyTorch will use {num_threads} threads for computation.")
    except NotImplementedError:
        print("Could not determine number of CPUs. Using PyTorch default.")

    # --- 1. 参数定义 ---
    CHUNK_SIZE = 2048
    NUM_SAMPLES = 20000 
    SEQUENCE_LENGTH = 100 * CHUNK_SIZE # 204800
    NUM_BINS = 20
    
    print("Generating simulation data...")
    simulator = DataSimulator(seed=42)
    dataset = simulator.generate_dataset(
        num_samples=NUM_SAMPLES, model_type='random', complexity='high', 
        sequence_length=SEQUENCE_LENGTH, num_bins=NUM_BINS, add_variable_rates=True
    )
    
    print("Data generation complete.")

    train_samples, test_samples = train_test_split(dataset, test_size=0.2, random_state=42)
    X_train = np.array([d['binary_seq'] for d in train_samples])
    y_train = np.array([d['normalized_histogram'] for d in train_samples])
    
    train_dataset = PSMCDataset(X_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # --- 4. 定义模型、损失函数和优化器 ---
    model = TransformerPSMC(
        output_size=NUM_BINS,
        chunk_size=CHUNK_SIZE,
        embedding_dim=128,
        nhead=4,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout_rate=0.2
    )

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    print("Starting model training...")
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=30)
    print("Model training complete.")

    print("Evaluating model and plotting results...")
    plot_psmc_comparison(model, test_samples) 