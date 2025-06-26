import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import importlib.util

# --- 核心模块导入 ---
from data_simulator import DataSimulator
from hybrid_model import HybridCNNBiLSTM
from transformer_hybrid_model import HybridCNNTransformer

# ==============================================================================
# 1. 动态模型导入器 和 辅助工具
# ==============================================================================

def import_model_from_file(file_path, class_name):
    """从指定文件动态导入一个类"""
    if not os.path.exists(file_path):
        print(f"警告: 模型文件 {file_path} 不存在。将跳过该模型。")
        return None
    spec = importlib.util.spec_from_file_location(name=f"model_module_{class_name}", location=file_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    return getattr(model_module, class_name)

class EarlyStopping:
    """早停机制，防止过拟合"""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping 计数: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'验证损失降低 ({self.val_loss_min:.6f} --> {val_loss:.6f})。正在保存模型 ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

# ==============================================================================
# 2. 数据集和核心评估流程
# ==============================================================================

class PopulationDataset(Dataset):
    """用于人口统计学模拟数据的PyTorch数据集类。"""
    def __init__(self, data, model_name):
        self.data = data
        self.model_name = model_name
        # MLP需要扁平化的输入
        self.flatten = 'MLP' in self.model_name
        # 原生的BiLSTM模型使用Embedding层，需要整数输入
        self.to_long = 'BiLSTM' in self.model_name and 'Hybrid' not in self.model_name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        seq = torch.from_numpy(sample['binary_seq'])
        # 根据模型需求转换数据类型
        seq = seq.long() if self.to_long else seq.float()
        hist = torch.from_numpy(sample['normalized_histogram']).float()
        if self.flatten:
            return seq.flatten(), hist
        return seq, hist

def collate_fn(batch, is_bilstm):
    """
    自定义的collate_fn来处理不同模型的数据类型需求。
    BiLSTM需要LongTensor，其他模型需要FloatTensor。
    """
    sequences, histograms = zip(*batch)
    
    # 对序列进行填充，以处理可能的长度不一致问题（尽管在这个脚本中长度是一致的）
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    histograms = torch.stack(histograms, 0)
    
    # 根据模型类型转换数据类型
    if is_bilstm:
        return sequences_padded.long(), histograms
    else:
        return sequences_padded.float(), histograms

def run_evaluation_for_model(model_name, model_config, train_loader, val_loader, device):
    """对单个模型进行训练、计时和评估，采用高级策略"""
    print(f"  实例化模型: {model_name}")
    model = model_config['class'](**model_config['params'])
    model.to(device)
    
    criterion = nn.KLDivLoss(reduction='batchmean')
    log_softmax = nn.LogSoftmax(dim=1)
    optimizer = optim.Adam(model.parameters(), lr=model_config['lr'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, verbose=True)
    
    checkpoint_path = f'best_{model_name.lower().replace("-", "_")}.pth'
    early_stopping = EarlyStopping(patience=5, verbose=False, path=checkpoint_path)

    print(f"  参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    start_time = time.time()
    best_epoch = 0
    
    for epoch in range(model_config['epochs']):
        # --- 训练 ---
        model.train()
        for sequences, histograms in train_loader:
            sequences, histograms = sequences.to(device), histograms.to(device)
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(log_softmax(outputs), histograms)
            loss.backward()
            optimizer.step()

        # --- 验证 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, histograms in val_loader:
                sequences, histograms = sequences.to(device), histograms.to(device)
                outputs = model(sequences)
                val_loss += criterion(log_softmax(outputs), histograms).item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"  周期 [{epoch+1}/{model_config['epochs']}], 验证损失: {avg_val_loss:.6f}")
        
        scheduler.step(avg_val_loss)
        early_stopping(avg_val_loss, model)
        
        if early_stopping.best_score is not None and -avg_val_loss == early_stopping.best_score:
            best_epoch = epoch + 1
            
        if early_stopping.early_stop:
            print("  检测到早停信号，终止训练。")
            break

    training_time = time.time() - start_time
    
    print(f"  训练耗时: {training_time:.2f}s | 最佳验证损失: {early_stopping.val_loss_min:.6f} at epoch {best_epoch}")
    
    return {
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "train_time_s": training_time,
        "best_val_loss": early_stopping.val_loss_min,
        "best_epoch": best_epoch,
        "stopped_early": early_stopping.early_stop
    }

# ==============================================================================
# 3. 主执行函数
# ==============================================================================

if __name__ == '__main__':
    # --- A. 全局配置 ---
    print("--- 1. 全局配置 ---")
    NUM_TRAIN_SAMPLES = 10000
    NUM_VAL_SAMPLES = 100
    SEQUENCE_LENGTH = 200000
    NUM_BINS = 20
    BATCH_SIZE = 64
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 充分利用CPU并行数据加载能力
    NUM_WORKERS = os.cpu_count()
    print(f"使用设备: {device}")
    print(f"使用 {NUM_WORKERS} 个worker进程进行数据加载。")

    # --- B. 定义模型配置 ---
    # 根据对每个模型文件的分析，精确配置超参数
    model_configurations = {
        "MLP": {
            "file": "mlp_model.py", "class_name": "MLP",
            "params": {
                "input_size": SEQUENCE_LENGTH, 
                "hidden_size": 2048, # 根据文件中的示例进行调整
                "output_size": NUM_BINS
            },
            "lr": 0.001, "epochs": 25
        },
        "CNN": {
            "file": "cnn_model.py", "class_name": "CNN",
            "params": {
                "output_size": NUM_BINS
            },
            "lr": 0.0005, "epochs": 30
        },
        "BiLSTM": {
            "file": "bilstm_model.py", "class_name": "BiLSTM",
            "params": {
                "vocab_size": 2, # 0 和 1
                "embedding_dim": 16,
                "hidden_size": 128,
                "output_size": NUM_BINS,
                "num_layers": 2
            },
            "lr": 0.0005, "epochs": 30
        },
        "Hybrid-CNN-BiLSTM": {
            "file": "hybrid_model.py", "class_name": "HybridCNNBiLSTM",
            "params": {
                "input_length": SEQUENCE_LENGTH, 
                "output_bins": NUM_BINS, 
                "cnn_out_channels": 32, 
                "lstm_hidden_size": 64
            },
            "lr": 0.0003, "epochs": 40
        },
        "Hybrid-CNN-Transformer": {
            "file": "transformer_hybrid_model.py", "class_name": "HybridCNNTransformer",
            "params": {
                "input_length": SEQUENCE_LENGTH, 
                "output_bins": NUM_BINS, 
                "cnn_out_channels": 64, 
                "transformer_d_model": 64, 
                "transformer_nhead": 8, 
                "transformer_num_layers": 4
            },
            "lr": 0.0001, "epochs": 50
        }
    }
    
    # --- C. 动态加载模型类 ---
    for name, config in model_configurations.items():
        model_class = import_model_from_file(config['file'], config['class_name'])
        if model_class:
            model_configurations[name]['class'] = model_class
    
    # 过滤掉加载失败的模型
    models_to_evaluate = {k: v for k, v in model_configurations.items() if 'class' in v}

    # --- D. 生成标准数据集 ---
    print("\n--- 2. 生成标准数据集 ---")
    simulator = DataSimulator(seed=123)
    train_data = simulator.generate_diverse_dataset(NUM_TRAIN_SAMPLES, sequence_length=SEQUENCE_LENGTH, num_bins=NUM_BINS)
    val_data = simulator.generate_diverse_dataset(NUM_VAL_SAMPLES, sequence_length=SEQUENCE_LENGTH, num_bins=NUM_BINS)
    
    # --- E. 循环评估所有模型 ---
    print("\n--- 3. 开始模型评估 ---")
    results = {}
    for name, config in models_to_evaluate.items():
        print(f"\n--- 正在评估模型: {name} ---")
        # 为每个模型创建独立的数据加载器
        train_dataset = PopulationDataset(train_data, model_name=name)
        val_dataset = PopulationDataset(val_data, model_name=name)
        # BiLSTM 使用 Embedding，需要整数输入
        is_bilstm = 'BiLSTM' in name and 'Hybrid' not in name
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, 
                                  collate_fn=lambda b: collate_fn(b, is_bilstm), pin_memory=device == 'cuda')
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                                collate_fn=lambda b: collate_fn(b, is_bilstm), pin_memory=device == 'cuda')
        
        results[name] = run_evaluation_for_model(name, config, train_loader, val_loader, device)
        
    # --- F. 结果汇总与展示 ---
    print("\n\n--- 4. 模型评估最终报告 ---")
    if not results:
        print("没有成功评估任何模型。请检查模型文件是否存在。")
    else:
        results_df = pd.DataFrame.from_dict(results, orient='index')
        results_df = results_df.sort_values(by='best_val_loss', ascending=True)
        
        results_df['params'] = results_df['params'].apply(lambda x: f"{x:,}")
        results_df['train_time_s'] = results_df['train_time_s'].apply(lambda x: f"{x:.2f}")
        results_df['best_val_loss'] = results_df['best_val_loss'].apply(lambda x: f"{x:.6f}")
        
        print(results_df)

        # 可视化结果
        results_df_numeric = pd.DataFrame.from_dict(results, orient='index')
        results_df_numeric.sort_values(by='best_val_loss', ascending=False, inplace=True)
        
        fig, ax = plt.subplots(1, 2, figsize=(18, 7))
        fig.suptitle('模型性能综合评估', fontsize=16)
        
        results_df_numeric['best_val_loss'].plot(kind='barh', ax=ax[0], color='c')
        ax[0].set_title('最佳验证损失 (越低越好)')
        ax[0].set_xlabel('KL散度损失')
        
        results_df_numeric['train_time_s'].plot(kind='barh', ax=ax[1], color='m')
        ax[1].set_title('总训练时间')
        ax[1].set_xlabel('秒 (s)')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show() 