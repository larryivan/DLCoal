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

class CNN(nn.Module):
    def __init__(self, output_size, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            # Block 2
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            # Block 3
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            # Block 4
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1, padding=4),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
        )
        
        # 为了动态计算fc1的输入大小，我们需要一个虚拟输入
        # 这里的sequence_length只是一个占位符，实际值在主程序中定义
        self._dummy_sequence_length = 500000 
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self._dummy_sequence_length)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(1, -1).size(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, output_size),
        )
        
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # Add a channel dimension: (N, L) -> (N, 1, L)
        x = x.unsqueeze(1)
        x = self.features(x)
        # Flatten the features
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return self.log_softmax(x)

def train_model(model, dataloader, criterion, optimizer, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')

def plot_psmc_comparison(model, test_samples, num_plots=5):
    """
    评估模型，并将预测的Ne历史与真实的Ne历史绘制成PSMC风格的对比图。
    """
    model.eval()
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = f"plots_cnn_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plots will be saved to '{plot_dir}/'")

    num_to_plot = min(num_plots, len(test_samples))
    
    for i in range(num_to_plot):
        sample = test_samples[i]
        input_seq = torch.from_numpy(sample['binary_seq']).unsqueeze(0)
        
        with torch.no_grad():
            predicted_log_dist = model(input_seq.float())
        
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
        plt.savefig(os.path.join(plot_dir, f"comparison_sample_{i+1}.png"), dpi=300)
        plt.close(fig)

if __name__ == '__main__':
    # --- 0. 配置 ---
    try:
        num_threads = int(multiprocessing.cpu_count()/2)
        torch.set_num_threads(num_threads)
        print(f"PyTorch will use {num_threads} threads for computation.")
    except NotImplementedError:
        print("Could not determine number of CPUs. Using PyTorch default.")

    # --- 1. 参数定义 ---
    NUM_SAMPLES = 500000
    SEQUENCE_LENGTH = 500000
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
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # --- 4. 定义模型、损失函数和优化器 ---
    output_size = NUM_BINS

    model = CNN(output_size=output_size, dropout_rate=0.5)
    
    # 动态调整CNN输入大小
    if model._dummy_sequence_length != SEQUENCE_LENGTH:
        print("Re-calculating classifier input size for new sequence length...")
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, SEQUENCE_LENGTH)
            dummy_output = model.features(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)
        model.classifier[0] = nn.Linear(flattened_size, 1024)
        print(f"New classifier input size: {flattened_size}")

    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting model training...")
    # --- 5. 训练模型 ---
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=50)
    print("Model training complete.")

    # --- 6. 评估并可视化结果 ---
    print("Evaluating model and plotting results...")
    plot_psmc_comparison(model, test_samples) 