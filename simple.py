import torch
import torch.nn as nn
import numpy as np
import math
import msprime
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import time
import os

class SimpleCoalNN(nn.Module):
    def __init__(self, focus_input_size, input_features=2, hidden_dims=[32, 64, 128], 
                 kernel_sizes=[5, 3, 3], root_time=float('inf')):
        super().__init__()
        self.focus_input_size = focus_input_size
        self.root_time = root_time
        
        layers = OrderedDict()
        
        # 输入归一化
        layers['norm'] = nn.BatchNorm1d(num_features=input_features)
        
        # 第一层使用膨胀卷积以增大感受野
        dilation = math.ceil(focus_input_size / (2 * kernel_sizes[0]))
        
        # 计算需要的填充以保持输出大小
        padding = ((kernel_sizes[0] - 1) * dilation) // 2
        
        # 第一个卷积层
        layers['conv1'] = nn.Conv1d(input_features, hidden_dims[0],
                                   kernel_size=kernel_sizes[0],
                                   padding=padding,  # 添加填充以保持序列长度
                                   dilation=dilation)
        layers['norm1'] = nn.BatchNorm1d(num_features=hidden_dims[0])
        layers['relu1'] = nn.ReLU()
        
        # 添加其余卷积层
        for i in range(1, len(hidden_dims)):
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else 3
            # 计算这一层需要的填充
            padding = (kernel_size - 1) // 2
            
            layers[f'conv{i+1}'] = nn.Conv1d(hidden_dims[i-1], hidden_dims[i],
                                            kernel_size=kernel_size,
                                            padding=padding,  # 添加填充以保持序列长度
                                            dilation=1)
            layers[f'norm{i+1}'] = nn.BatchNorm1d(num_features=hidden_dims[i])
            layers[f'relu{i+1}'] = nn.ReLU()
        
        # 输出层 - 只输出TMRCA预测（一个通道）
        layers[f'conv{len(hidden_dims)+1}'] = nn.Conv1d(hidden_dims[-1], 1, 
                                                       kernel_size=1, 
                                                       dilation=1)
        
        self.network = nn.Sequential(layers)
        
    def forward(self, x):
        # 如果输入是字典格式，则提取'input'键
        if isinstance(x, dict):
            x = x['input']
            
        # 前向传播
        prediction = self.network(x)
        
        # 限制输出范围
        output = torch.clamp(prediction[:, 0, :], min=math.log(0.01), max=self.root_time)
        
        return output

def generate_coalescent_data(num_samples, sequence_length, Ne=10000, mutation_rate=1e-8, recombination_rate=1e-8):
    """
    使用msprime生成模拟数据 - CoalNN风格的增强版
    
    参数:
    num_samples: 要生成的样本对数
    sequence_length: 每个样本的序列长度
    Ne: 有效群体大小
    mutation_rate: 突变率
    recombination_rate: 重组率
    
    返回:
    haplotypes: 单倍型数据，形状为[num_samples, 2, sequence_length]
    tmrca: 对应的TMRCA值，形状为[num_samples, sequence_length]
    """
    all_haplotypes = []
    all_tmrca = []
    
    # 创建不同的人口历史场景
    demographic_models = [
        # 常数大小人口
        msprime.Demography(),
        
        # 人口扩张
        msprime.Demography.isolated_model([1.0], 
            growth_rate=[0.01],
            initial_size=[Ne]),
            
        # 人口瓶颈
        msprime.Demography.population_split(time=0.1*Ne, 
            initial_size=[Ne*0.5, Ne], 
            growth_rate=[0, 0], 
            demographics_id="bottleneck")
    ]
    
    # 为每个样本随机选择不同的参数
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"生成样本 {i}/{num_samples}")
        
        # 随机选择人口模型
        demog_model = np.random.choice(demographic_models)
        
        # 随机调整变异率和重组率（增加多样性）
        mut_rate = mutation_rate * np.random.uniform(0.5, 2.0)
        rec_rate = recombination_rate * np.random.uniform(0.5, 2.0)
        
        # 随机有效群体大小
        effective_size = Ne * np.random.uniform(0.8, 1.2)
        
        # 生成树序列
        ts = msprime.simulate(
            sample_size=2,  # 每次只考虑两条序列
            Ne=effective_size,
            length=sequence_length,
            mutation_rate=mut_rate,
            recombination_rate=rec_rate,
            record_full_arg=True,
            # 如果选择了人口模型，则使用它
            demography=demog_model if demog_model.populations else None 
        )
        
        # 提取单倍型
        haplotypes = np.zeros((2, sequence_length), dtype=np.int8)
        for variant in ts.variants():
            pos = int(variant.position)
            if pos < sequence_length:
                haplotypes[:, pos] = variant.genotypes
        
        # 提取TMRCA
        tmrca = np.zeros(sequence_length)
        for tree in ts.trees():
            left, right = int(tree.interval.left), int(min(tree.interval.right, sequence_length))
            tmrca[left:right] = tree.tmrca(0, 1) * 2 * effective_size  # 转换为2Ne单位
        
        all_haplotypes.append(haplotypes)
        all_tmrca.append(tmrca)
    
    return np.array(all_haplotypes), np.array(all_tmrca)

class CoalescentDataset(Dataset):
    def __init__(self, haplotypes, tmrca, log_transform=True, window_size=20):
        self.haplotypes = haplotypes
        self.tmrca = tmrca
        self.log_transform = log_transform
        self.window_size = window_size
        
    def __len__(self):
        return len(self.haplotypes)
    
    def __getitem__(self, idx):
        haplotype_pair = self.haplotypes[idx]
        tmrca_values = self.tmrca[idx]
        
        # 基础特征: XOR和AND
        xor_feature = np.logical_xor(haplotype_pair[0], haplotype_pair[1]).astype(np.float32)
        and_feature = np.logical_and(haplotype_pair[0], haplotype_pair[1]).astype(np.float32)
        
        # 特征1-2: 基础XOR和AND特征 (保留)
        
        # 特征3: 局部突变密度 (已有，但优化)
        # 使用指数加权窗口而非简单均值，让近处突变有更高权重
        seq_length = haplotype_pair.shape[1]
        local_diff_density = np.zeros(seq_length, dtype=np.float32)
        for pos in range(seq_length):
            weights_sum = 0
            weighted_diff_sum = 0
            for offset in range(-self.window_size, self.window_size + 1):
                if 0 <= pos + offset < seq_length:
                    # 指数衰减权重
                    weight = np.exp(-abs(offset) / (self.window_size / 3))
                    weights_sum += weight
                    weighted_diff_sum += weight * xor_feature[pos + offset]
            local_diff_density[pos] = weighted_diff_sum / weights_sum if weights_sum > 0 else 0
        
        # 特征4: 距离上一个差异的位置 (已有)
        dist_to_diff = np.zeros(seq_length, dtype=np.float32)
        last_diff_pos = -1
        for pos in range(seq_length):
            if xor_feature[pos] == 1:
                last_diff_pos = pos
            if last_diff_pos >= 0:
                dist_to_diff[pos] = (pos - last_diff_pos) / seq_length  # 归一化距离
            else:
                dist_to_diff[pos] = 1.0  # 如果之前没有差异点
        
        # 特征5: 距离下一个差异的位置 (新特征)
        dist_to_next_diff = np.zeros(seq_length, dtype=np.float32)
        last_diff_pos = seq_length  # 初始化为序列末尾
        for pos in range(seq_length-1, -1, -1):  # 从右向左
            if xor_feature[pos] == 1:
                last_diff_pos = pos
            if last_diff_pos < seq_length:
                dist_to_next_diff[pos] = (last_diff_pos - pos) / seq_length  # 归一化距离
            else:
                dist_to_next_diff[pos] = 1.0  # 如果右侧没有差异点
        
        # 特征6: 局部突变模式 - 输出LD信息 (新特征)
        # 在小窗口内计算成对LD
        ld_pattern = np.zeros(seq_length, dtype=np.float32)
        for pos in range(seq_length):
            # 找到附近的突变
            start = max(0, pos - self.window_size)
            end = min(seq_length, pos + self.window_size + 1)
            
            # 统计窗口内突变位点
            window_variants = np.where(xor_feature[start:end] > 0)[0]
            if len(window_variants) >= 2:  # 需要至少2个变异位点才能计算LD
                # 简单LD得分 - 相邻变异位点的平均距离
                dists = window_variants[1:] - window_variants[:-1]
                if len(dists) > 0:
                    ld_pattern[pos] = np.mean(dists) / self.window_size  # 归一化
        
        # 特征7: 全局位置特征 - 相对位置信息
        rel_position = np.linspace(0, 1, seq_length, dtype=np.float32)
        
        # 组合所有特征
        features = np.stack([
            xor_feature,           # 特征1: 基本突变模式
            and_feature,           # 特征2: 共享等位基因
            local_diff_density,    # 特征3: 增强的局部突变密度
            dist_to_diff,          # 特征4: 距离上一个突变
            dist_to_next_diff,     # 特征5: 距离下一个突变
            ld_pattern,            # 特征6: 局部连锁不平衡特征
            rel_position           # 特征7: 相对位置特征
        ], axis=0)
        
        # 转换为torch tensor
        input_tensor = torch.tensor(features).float()
        
        # 处理目标值
        if self.log_transform:
            # 确保TMRCA值为正，然后取对数
            tmrca_values = np.maximum(tmrca_values, 0.01)  # 防止取log(0)
            target = torch.tensor(np.log(tmrca_values)).float()
        else:
            target = torch.tensor(tmrca_values).float()
        
        return {'input': input_tensor, 'target': target}

def prepare_data(haplotypes, add_features=None, window_size=20):
    """
    准备模型输入数据 
    
    参数:
    haplotypes: numpy数组，形状为[2, sequence_length]，表示两条单倍型序列
    add_features: 其他特征，如遗传距离、物理距离等，形状为[n_features, sequence_length]
    window_size: 计算局部特征的窗口大小
    
    返回:
    torch.Tensor，形状为[1, num_features, sequence_length]
    """
    # 计算基础特征
    h1 = haplotypes[0]
    h2 = haplotypes[1]
    seq_length = len(h1)
    
    xor_feature = np.logical_xor(h1, h2).astype(np.float32)
    and_feature = np.logical_and(h1, h2).astype(np.float32)
    
    # 特征1-2: 基础XOR和AND特征 (保留)
    
    # 特征3: 增强的局部突变密度 - 指数加权
    local_diff_density = np.zeros(seq_length, dtype=np.float32)
    for pos in range(seq_length):
        weights_sum = 0
        weighted_diff_sum = 0
        for offset in range(-window_size, window_size + 1):
            if 0 <= pos + offset < seq_length:
                # 指数衰减权重
                weight = np.exp(-abs(offset) / (window_size / 3))
                weights_sum += weight
                weighted_diff_sum += weight * xor_feature[pos + offset]
        local_diff_density[pos] = weighted_diff_sum / weights_sum if weights_sum > 0 else 0
    
    # 特征4: 距离上一个差异的位置
    dist_to_diff = np.zeros(seq_length, dtype=np.float32)
    last_diff_pos = -1
    for pos in range(seq_length):
        if xor_feature[pos] == 1:
            last_diff_pos = pos
        if last_diff_pos >= 0:
            dist_to_diff[pos] = (pos - last_diff_pos) / seq_length  # 归一化距离
        else:
            dist_to_diff[pos] = 1.0  # 如果之前没有差异点
    
    # 特征5: 距离下一个差异的位置
    dist_to_next_diff = np.zeros(seq_length, dtype=np.float32)
    last_diff_pos = seq_length  # 初始化为序列末尾
    for pos in range(seq_length-1, -1, -1):  # 从右向左
        if xor_feature[pos] == 1:
            last_diff_pos = pos
        if last_diff_pos < seq_length:
            dist_to_next_diff[pos] = (last_diff_pos - pos) / seq_length  # 归一化距离
        else:
            dist_to_next_diff[pos] = 1.0  # 如果右侧没有差异点
    
    # 特征6: 局部突变模式 - LD信息
    ld_pattern = np.zeros(seq_length, dtype=np.float32)
    for pos in range(seq_length):
        # 找到附近的突变
        start = max(0, pos - window_size)
        end = min(seq_length, pos + window_size + 1)
        
        # 统计窗口内突变位点
        window_variants = np.where(xor_feature[start:end] > 0)[0] + start
        if len(window_variants) >= 2:  # 需要至少2个变异位点才能计算LD
            # 简单LD得分 - 相邻变异位点的平均距离
            dists = window_variants[1:] - window_variants[:-1]
            if len(dists) > 0:
                ld_pattern[pos] = np.mean(dists) / window_size  # 归一化
    
    # 特征7: 全局位置特征 - 相对位置信息
    rel_position = np.linspace(0, 1, seq_length, dtype=np.float32)
    
    # 组合所有特征
    features = np.stack([
        xor_feature,           # 基本突变模式
        and_feature,           # 共享等位基因
        local_diff_density,    # 增强的局部突变密度
        dist_to_diff,          # 距离上一个突变
        dist_to_next_diff,     # 距离下一个突变
        ld_pattern,            # 局部连锁不平衡特征
        rel_position           # 相对位置特征
    ], axis=0)
    
    # 添加额外特征
    if add_features is not None:
        features = np.concatenate([features, add_features], axis=0)
    
    # 转换为torch tensor并添加批次维度
    input_tensor = torch.tensor(features).unsqueeze(0).float()
    
    return input_tensor

class ResidualBlock(nn.Module):
    """残差块 - 受CoalNN启发"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = (kernel_size - 1) // 2  # 相同填充
        
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 添加残差连接
        out = self.relu(out)
        return out

class ImprovedCoalNN(nn.Module):
    """
    增加了残差连接和更多高级特征
    """
    def __init__(self, focus_input_size, input_features=4, hidden_dims=[64, 128, 256], 
                 kernel_sizes=[7, 5, 3], num_residual_blocks=2, root_time=float('inf')):
        super().__init__()
        self.focus_input_size = focus_input_size
        self.root_time = root_time
        
        layers = OrderedDict()
        
        # 输入归一化
        layers['norm'] = nn.BatchNorm1d(num_features=input_features)
        
        # 第一层使用膨胀卷积以增大感受野
        dilation = max(1, math.ceil(focus_input_size / (2 * kernel_sizes[0])))
        padding = ((kernel_sizes[0] - 1) * dilation) // 2
        
        # 第一个卷积层
        layers['conv1'] = nn.Conv1d(input_features, hidden_dims[0],
                                  kernel_size=kernel_sizes[0],
                                  padding=padding,
                                  dilation=dilation)
        layers['norm1'] = nn.BatchNorm1d(num_features=hidden_dims[0])
        layers['relu1'] = nn.ReLU()
        
        # 添加残差块
        for i in range(num_residual_blocks):
            layers[f'residual{i+1}'] = ResidualBlock(hidden_dims[0], kernel_size=3)
        
        # 添加其余卷积层
        for i in range(1, len(hidden_dims)):
            kernel_size = kernel_sizes[i] if i < len(kernel_sizes) else 3
            padding = (kernel_size - 1) // 2
            
            layers[f'conv{i+1}'] = nn.Conv1d(hidden_dims[i-1], hidden_dims[i],
                                           kernel_size=kernel_size,
                                           padding=padding,
                                           dilation=1)
            layers[f'norm{i+1}'] = nn.BatchNorm1d(num_features=hidden_dims[i])
            layers[f'relu{i+1}'] = nn.ReLU()
            
            # 每层后增加残差块
            for j in range(num_residual_blocks):
                layers[f'residual{i+1}_{j+1}'] = ResidualBlock(hidden_dims[i], kernel_size=3)
        
        # 输出层 - 只输出TMRCA预测（一个通道）
        layers[f'conv{len(hidden_dims)+1}'] = nn.Conv1d(hidden_dims[-1], 1, 
                                                      kernel_size=1, 
                                                      dilation=1)
        
        self.network = nn.Sequential(layers)
        
    def forward(self, x):
        # 如果输入是字典格式，则提取'input'键
        if isinstance(x, dict):
            x = x['input']
            
        # 前向传播
        prediction = self.network(x)
        
        # 限制输出范围
        output = torch.clamp(prediction[:, 0, :], min=math.log(0.01), max=self.root_time)
        
        return output

if __name__ == "__main__":
    # 设置随机种子以便结果可重复
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检测CUDA是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 数据生成参数 - 使用大量数据
    train_samples = 50000  # 5万个样本用于训练
    val_samples = 5000    # 5千个样本用于验证
    test_samples = 2000   # 2千个样本用于测试
    sequence_length = 1000  # 每个序列1000个位点
    
    # 检查是否已经有保存的数据
    data_file = 'improved_coalescent_data.npz'
    if os.path.exists(data_file):
        print(f'加载已存在的数据 {data_file}...')
        data = np.load(data_file)
        train_haplotypes = data['train_haplotypes']
        train_tmrca = data['train_tmrca']
        val_haplotypes = data['val_haplotypes']
        val_tmrca = data['val_tmrca']
        test_haplotypes = data['test_haplotypes']
        test_tmrca = data['test_tmrca']
    else:
        # 生成数据
        print('正在生成训练数据...')
        train_haplotypes, train_tmrca = generate_coalescent_data(train_samples, sequence_length)
        
        print('正在生成验证数据...')
        val_haplotypes, val_tmrca = generate_coalescent_data(val_samples, sequence_length)
        
        print('正在生成测试数据...')
        test_haplotypes, test_tmrca = generate_coalescent_data(test_samples, sequence_length)
        
        # 保存数据以便重用
        print('保存生成的数据...')
        np.savez(data_file, 
                train_haplotypes=train_haplotypes, 
                train_tmrca=train_tmrca,
                val_haplotypes=val_haplotypes, 
                val_tmrca=val_tmrca,
                test_haplotypes=test_haplotypes,
                test_tmrca=test_tmrca)
    
    print(f'训练样本: {train_haplotypes.shape[0]}')
    print(f'验证样本: {val_haplotypes.shape[0]}')
    print(f'测试样本: {test_haplotypes.shape[0]}')
    
    # 创建数据集
    window_size = 20  # 局部特征的窗口大小
    train_dataset = CoalescentDataset(train_haplotypes, train_tmrca, window_size=window_size)
    val_dataset = CoalescentDataset(val_haplotypes, val_tmrca, window_size=window_size)
    test_dataset = CoalescentDataset(test_haplotypes, test_tmrca, window_size=window_size)
    
    # 创建数据加载器
    batch_size = 128
    num_workers = min(4, os.cpu_count() or 1)  # 避免使用过多worker
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    
    # 检查是否有保存的模型
    model_path = 'improved_coalnn_model.pt'
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        print(f'加载已训练模型 {model_path}')
        checkpoint = torch.load(model_path, map_location=device)
        model = ImprovedCoalNN(focus_input_size=sequence_length,
                               input_features=4,  # 现在使用4个特征
                               hidden_dims=[64, 128, 256, 512],
                               kernel_sizes=[7, 5, 3, 3],
                               num_residual_blocks=2)
        model.load_state_dict(checkpoint['model'])
    else:
        # 创建模型
        model = ImprovedCoalNN(focus_input_size=sequence_length,
                               input_features=4,  # 现在使用4个特征
                               hidden_dims=[64, 128, 256, 512],
                               kernel_sizes=[7, 5, 3, 3],
                               num_residual_blocks=2)
        
        # 训练模型
        print('开始训练改进版CoalNN模型...')
        num_epochs = 50
        model, history, best_model = train_model(model, train_loader, val_loader, num_epochs, learning_rate=0.001, device=device)
        
        # 保存模型
        torch.save(best_model, model_path)
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.savefig('improved_training_loss.png', dpi=300)
        plt.show()
    
    # 在测试集上评估模型
    print('在测试集上评估改进版模型...')
    predictions, targets = evaluate_model(model, test_loader, device=device)
    
    # 可视化结果
    print('可视化结果...')
    mse, mae, r2 = plot_results(predictions, targets, log_transform=True)