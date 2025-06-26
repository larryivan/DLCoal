import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

from data_simulator import DataSimulator
from hybrid_model import HybridCNNBiLSTM

# --- 1. 自定义数据集类 ---
class PopulationDataset(Dataset):
    """
    用于人口统计学模拟数据的PyTorch数据集类。
    """
    def __init__(self, data):
        """
        初始化数据集。

        参数:
            data (list): 由DataSimulator生成的样本列表。
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取单个样本。

        返回:
            tuple: (二进制序列, 归一化的直方图)
        """
        sample = self.data[idx]
        # 输入：二进制序列
        binary_seq = torch.from_numpy(sample['binary_seq']).float()
        # 目标：归一化的直方图
        norm_hist = torch.from_numpy(sample['normalized_histogram']).float()
        return binary_seq, norm_hist

# --- 2. 训练和评估函数 ---
def train_model(model, train_loader, val_loader, num_epochs=25, learning_rate=1e-4, device='cpu'):
    """
    训练和评估模型。

    参数:
        model (nn.Module): 要训练的模型。
        train_loader (DataLoader): 训练数据加载器。
        val_loader (DataLoader): 验证数据加载器。
        num_epochs (int): 训练周期数。
        learning_rate (float): 学习率。
        device (str): 训练设备 ('cpu' 或 'cuda')。

    返回:
        dict: 包含训练和验证损失历史的字典。
    """
    # 将模型移动到指定设备
    model.to(device)

    # 损失函数：Kullback-Leibler Divergence Loss
    # 适用于比较两个概率分布。模型输出logits，目标是概率分布。
    # LogSoftmax将logits转换为log-probabilities。
    criterion = nn.KLDivLoss(reduction='batchmean')
    log_softmax = nn.LogSoftmax(dim=1)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 用于记录损失历史
    history = {'train_loss': [], 'val_loss': []}

    print("\\n--- 开始训练 ---")
    for epoch in range(num_epochs):
        # --- 训练阶段 ---
        model.train()
        running_train_loss = 0.0
        for i, (sequences, histograms) in enumerate(train_loader):
            sequences, histograms = sequences.to(device), histograms.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # 前向传播
            outputs = model(sequences)
            log_probs = log_softmax(outputs) # 计算log-probabilities
            
            # 计算损失
            loss = criterion(log_probs, histograms)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()

        epoch_train_loss = running_train_loss / len(train_loader)
        history['train_loss'].append(epoch_train_loss)

        # --- 验证阶段 ---
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for sequences, histograms in val_loader:
                sequences, histograms = sequences.to(device), histograms.to(device)
                outputs = model(sequences)
                log_probs = log_softmax(outputs)
                loss = criterion(log_probs, histograms)
                running_val_loss += loss.item()
        
        epoch_val_loss = running_val_loss / len(val_loader)
        history['val_loss'].append(epoch_val_loss)

        print(f"周期 [{epoch+1}/{num_epochs}], "
              f"训练损失: {epoch_train_loss:.6f}, "
              f"验证损失: {epoch_val_loss:.6f}")

    print("--- 训练完成 ---")
    return history

def plot_loss(history):
    """
    绘制训练和验证损失曲线。
    """
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='训练损失 (Train Loss)')
    plt.plot(history['val_loss'], label='验证损失 (Validation Loss)')
    plt.title('训练和验证损失曲线')
    plt.xlabel('周期 (Epoch)')
    plt.ylabel('损失 (Loss)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # --- 3. 主执行流程 ---

    # A. 参数设置
    print("--- 1. 参数设置 ---")
    NUM_TRAIN_SAMPLES = 200   # 训练样本数量 (为演示目的设置较小值)
    NUM_VAL_SAMPLES = 50      # 验证样本数量
    SEQUENCE_LENGTH = 10000   # 序列长度 (为快速演示设置较小值)
    NUM_BINS = 20             # 直方图的区间数
    BATCH_SIZE = 16
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.0005
    MODEL_SAVE_PATH = 'hybrid_cnn_bilstm_model.pth'

    # 检查是否有可用的GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # B. 数据生成
    print("\\n--- 2. 生成数据集 ---")
    # 使用固定的种子以保证数据可复现
    simulator = DataSimulator(seed=42)
    
    print(f"正在生成 {NUM_TRAIN_SAMPLES} 个训练样本...")
    train_data = simulator.generate_diverse_dataset(
        num_samples=NUM_TRAIN_SAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        num_bins=NUM_BINS
    )
    
    print(f"正在生成 {NUM_VAL_SAMPLES} 个验证样本...")
    val_data = simulator.generate_diverse_dataset(
        num_samples=NUM_VAL_SAMPLES,
        sequence_length=SEQUENCE_LENGTH,
        num_bins=NUM_BINS
    )

    # C. 数据加载
    print("\\n--- 3. 创建数据加载器 ---")
    train_dataset = PopulationDataset(train_data)
    val_dataset = PopulationDataset(val_data)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("数据加载器创建完成。")

    # D. 模型初始化
    print("\\n--- 4. 初始化模型 ---")
    model = HybridCNNBiLSTM(input_length=SEQUENCE_LENGTH, output_bins=NUM_BINS)
    print(f"模型总参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # E. 模型训练
    training_history = train_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=NUM_EPOCHS, 
        learning_rate=LEARNING_RATE,
        device=device
    )

    # F. 结果可视化
    print("\\n--- 5. 绘制损失曲线 ---")
    plot_loss(training_history)

    # G. 模型保存
    print(f"\\n--- 6. 保存训练好的模型到 {MODEL_SAVE_PATH} ---")
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print("模型已保存。") 