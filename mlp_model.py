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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 8, hidden_size // 16),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 16, output_size)
        )
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layers(x)
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
    
    # 创建一个带时间戳的目录来保存图片
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_dir = f"plots_{timestamp}"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Plots will be saved to '{plot_dir}/'")

    # 从测试集中选择几个样本进行绘图
    num_to_plot = min(num_plots, len(test_samples))
    
    for i in range(num_to_plot):
        sample = test_samples[i]
        input_seq = torch.from_numpy(sample['binary_seq']).unsqueeze(0)
        
        with torch.no_grad():
            predicted_log_dist = model(input_seq.float())
        
        predicted_dist = torch.exp(predicted_log_dist).numpy().flatten()
        
        # 准备绘图
        fig, ax = plt.subplots(figsize=(12, 7))

        # 1. 绘制真实的 demographic history
        true_times = [0]
        true_ne = [sample['Ne']]
        for event in sorted(sample['demographic_events'], key=lambda e: e.time):
            true_times.append(event.time)
            true_ne.append(event.initial_size)
        ax.step(true_times, true_ne, where='post', label='True Ne History', color='green', linewidth=2.5, zorder=2)
        
        # 2. 计算并绘制预测的Ne历史
        bin_edges = sample['bin_edges']
        bin_widths = np.diff(bin_edges)
        bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # 避免除以零
        # 概率密度与Ne成反比，所以Ne与 1/密度 成正比
        # Ne_pred ∝ 1 / (predicted_dist / bin_widths) = bin_widths / predicted_dist
        predicted_dist_nonzero = predicted_dist + 1e-12 # 防止除以0
        relative_ne_pred = bin_widths / predicted_dist_nonzero
        
        # 使用一个简单的缩放因子来对齐预测值和真实值
        # 这里我们让预测的Ne的均值与真实的Ne的均值（在绘图时间范围内）相匹配
        visible_true_ne = np.array(true_ne)[np.array(true_times) < bin_edges[-1]]
        if len(visible_true_ne) == 0:
             # 如果真实历史太久远，就用初始Ne
            mean_true_ne = sample['Ne']
        else:
            mean_true_ne = np.mean(visible_true_ne)

        scaling_factor = mean_true_ne / np.mean(relative_ne_pred)
        ne_pred = relative_ne_pred * scaling_factor

        ax.step(bin_midpoints, ne_pred, where='mid', label='Predicted Ne History', color='purple', linestyle='--', linewidth=2, zorder=3)
        
        # 设置坐标轴和标题
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Time (generations ago) [Log Scale]')
        ax.set_ylabel('Effective Population Size (Ne) [Log Scale]')
        ax.set_title(f"PSMC Comparison for Sample {i+1} (Model: {sample['model_type']})")
        ax.legend()
        ax.grid(True, which="both", ls="--", c='0.7', zorder=0)
        ax.set_xlim(left=max(10, bin_edges[1] * 0.5), right=bin_edges[-1] * 1.5)
        
        # 保存图片
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"comparison_sample_{i+1}.png"), dpi=300)
        plt.close(fig) # 关闭图形，防止在Jupyter等环境中直接显示

if __name__ == '__main__':
    # --- 0. 配置PyTorch以使用多CPU核心 ---
    try:
        num_threads = multiprocessing.cpu_count()
        torch.set_num_threads(num_threads)
        print(f"PyTorch will use {num_threads} threads for computation.")
    except NotImplementedError:
        print("Could not determine number of CPUs. Using PyTorch default.")

    # --- 1. 参数定义 ---
    NUM_SAMPLES = 100000
    SEQUENCE_LENGTH = 200000
    NUM_BINS = 20
    
    # --- 2. 使用DataSimulator生成数据 ---
    print("Generating simulation data...")
    simulator = DataSimulator(seed=42)
    # 使用复杂度较高的配置
    dataset = simulator.generate_dataset(
        num_samples=NUM_SAMPLES, 
        model_type='random', 
        complexity='high', 
        sequence_length=SEQUENCE_LENGTH, 
        num_bins=NUM_BINS,
        add_variable_rates=True # 启用可变速率以增加真实性
    )
    
    input_sequences = np.array([d['binary_seq'] for d in dataset])
    target_distributions = np.array([d['normalized_histogram'] for d in dataset])
    bin_edges = dataset[0]['bin_edges'] # 获取所有样本通用的时间窗口边界

    print("Data generation complete.")

    # --- 3. 创建数据集和数据加载器 ---
    train_samples, test_samples = train_test_split(dataset, test_size=0.2, random_state=42)

    X_train = np.array([d['binary_seq'] for d in train_samples])
    y_train = np.array([d['normalized_histogram'] for d in train_samples])
    X_test = np.array([d['binary_seq'] for d in test_samples])
    y_test = np.array([d['normalized_histogram'] for d in test_samples])

    train_dataset = PSMCDataset(X_train, y_train)
    test_dataset = PSMCDataset(X_test, y_test)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # --- 4. 定义模型、损失函数和优化器 ---
    input_size = SEQUENCE_LENGTH
    hidden_size = 4096
    output_size = NUM_BINS

    model = MLP(input_size, hidden_size, output_size, dropout_rate=0.5)
    criterion = nn.KLDivLoss(reduction='batchmean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Starting model training...")
    # --- 5. 训练模型 ---
    train_model(model, train_dataloader, criterion, optimizer, num_epochs=50)
    print("Model training complete.")

    # --- 6. 评估并可视化结果 ---
    print("Evaluating model and plotting results...")
    plot_psmc_comparison(model, test_samples) 