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

from data_simulator import DataSimulator

class PSMCDataset(Dataset):
    def __init__(self, sequences, tmrca_distributions):
        self.sequences = sequences
        self.tmrca_distributions = tmrca_distributions

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.tmrca_distributions[idx]

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        # scores shape: (batch_size, seq_len, 1)
        scores = self.attention(lstm_output)
        # weights shape: (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)
        # context shape: (batch_size, hidden_size)
        context = torch.sum(weights * lstm_output, dim=1)
        return context

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers, dropout_rate=0.5):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, 
                              hidden_size, 
                              num_layers=num_layers, 
                              bidirectional=True, 
                              batch_first=True,
                              dropout=dropout_rate if num_layers > 1 else 0)
        
        self.attention = Attention(hidden_size * 2) # *2 for bidirectional
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size)
        )
        
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Input x is already long type from dataloader, no need for .long()
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        
        context = self.attention(lstm_out)
        
        logits = self.classifier(context)
        return self.log_softmax(logits)

def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            # Input to LSTM should be LongTensor
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
    plot_dir = f"plots_bilstm_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plots will be saved to '{plot_dir}/'")

    num_to_plot = min(num_plots, len(test_samples))
    
    for i in range(num_to_plot):
        sample = test_samples[i]
        input_seq = torch.from_numpy(sample['binary_seq']).unsqueeze(0)
        
        with torch.no_grad():
            predicted_log_dist = model(input_seq.long())
        
        predicted_dist = torch.exp(predicted_log_dist).numpy().flatten()
        
        fig, ax = plt.subplots(figsize=(12, 7))

        true_times = [0]
        true_ne = [sample['Ne']]
        for event in sorted(sample['demographic_events'], key=lambda e: e.time):
            true_times.append(event.time)
            true_ne.append(event.initial_size)
        ax.step(true_times, true_ne, where='post', label='True Ne History', color='green', linewidth=2.5, zorder=2)
        
        bin_edges = sample['bin_edges']
        bin_widths = np.diff(bin_edges)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        predicted_dist_nonzero = predicted_dist + 1e-12
        relative_ne_pred = bin_widths / predicted_dist_nonzero
        
        visible_true_ne = np.array(true_ne)[np.array(true_times) < bin_edges[-1]]
        if len(visible_true_ne) == 0:
            mean_true_ne = sample['Ne']
        else:
            mean_true_ne = np.mean(visible_true_ne)

        scaling_factor = mean_true_ne / np.mean(relative_ne_pred)
        ne_pred = relative_ne_pred * scaling_factor

        ax.step(bin_midpoints, ne_pred, where='mid', label='Predicted Ne History', color='purple', linestyle='--', linewidth=2, zorder=3)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time (generations ago) [Log Scale]')
        ax.set_ylabel('Effective Population Size (Ne) [Log Scale]')
        ax.set_title(f"PSMC Comparison for Sample {i+1} (Model: {sample['model_type']})")
        ax.legend()
        ax.grid(True, which="both", ls="--", c='0.7', zorder=0)
        ax.set_xlim(left=max(10, bin_edges[1] * 0.5), right=bin_edges[-1] * 1.5)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"bilstm_comparison_sample_{i+1}.png"), dpi=300)
        plt.close(fig)

if __name__ == '__main__':
    try:
        num_threads = int(multiprocessing.cpu_count()/2)
        torch.set_num_threads(num_threads)
        print(f"PyTorch will use {num_threads} threads for computation.")
    except NotImplementedError:
        print("Could not determine number of CPUs. Using PyTorch default.")

    # --- 1. 参数定义 ---
    NUM_SAMPLES = 10000 # 减小规模以适应LSTM的计算需求
    SEQUENCE_LENGTH = 100000 # 减小规模以适应LSTM的计算需求
    NUM_BINS = 20
    
    # --- 2. 使用DataSimulator生成数据 ---
    print("Generating simulation data...")
    simulator = DataSimulator(seed=42)
    dataset = simulator.generate_dataset(
        num_samples=NUM_SAMPLES, 
        model_type='random', 
        complexity='high', 
        sequence_length=SEQUENCE_LENGTH, 
        num_bins=NUM_BINS,
        add_variable_rates=True
    )
    
    print("Data generation complete.")

    # --- 3. 创建数据集和数据加载器 ---
    train_samples, test_samples = train_test_split(dataset, test_size=0.2, random_state=42)

    X_train = np.array([d['binary_seq'] for d in train_samples])
    y_train = np.array([d['normalized_histogram'] for d in train_samples])
    
    train_dataset = PSMCDataset(X_train, y_train)
    
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True) # 减小batch size

    # --- 4. 定义模型、损失函数和优化器 ---
    model = BiLSTM(
        vocab_size=2, # 0 and 1
        embedding_dim=16,
        hidden_size=128,
        output_size=NUM_BINS,
        num_layers=2,
        dropout_rate=0.5
    )

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting model training...")
    # --- 5. 训练模型 ---
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=20) # 减少epoch
    print("Model training complete.")

    # --- 6. 评估并可视化结果 ---
    print("Evaluating model and plotting results...")
    plot_psmc_comparison(model, test_samples) 