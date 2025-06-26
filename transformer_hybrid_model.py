import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """
    为Transformer注入序列的位置信息。
    """
    def __init__(self, d_model, dropout=0.1, max_len=50000):
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
        """
        参数:
            x: 形状为 [seq_len, batch_size, d_model] 的张量
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class HybridCNNTransformer(nn.Module):
    """
    一个更先进的混合模型，结合了CNN和Transformer编码器。

    模型结构:
    1.  CNN层：用于高效提取局部特征并对序列进行降采样。
    2.  位置编码（Positional Encoding）：为序列注入位置信息，这是Transformer所必需的。
    3.  Transformer编码器层：利用自注意力机制捕捉全局和长距离依赖关系。
    4.  全连接层：将Transformer的输出映射到最终预测。
    """
    def __init__(self, input_length, output_bins, cnn_out_channels=32, kernel_size=11, 
                 transformer_d_model=32, transformer_nhead=4, transformer_num_layers=3):
        """
        初始化CNN-Transformer混合模型。

        参数:
            input_length (int): 输入序列的长度。
            output_bins (int): 输出的维度（例如，直方图的区间数）。
            cnn_out_channels (int): CNN层的输出通道数。
            kernel_size (int): CNN卷积核的大小。
            transformer_d_model (int): Transformer的特征维度 (必须等于cnn_out_channels)。
            transformer_nhead (int): Transformer多头注意力的头数。
            transformer_num_layers (int): Transformer编码器的层数。
        """
        super(HybridCNNTransformer, self).__init__()
        
        if cnn_out_channels != transformer_d_model:
            raise ValueError("cnn_out_channels必须等于transformer_d_model")

        # --- 1. CNN特征提取器 ---
        self.cnn_extractor = nn.Sequential(
            nn.Conv1d(1, cnn_out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.2)
        )
        
        # --- 2. 位置编码 ---
        self.pos_encoder = PositionalEncoding(d_model=transformer_d_model)

        # --- 3. Transformer编码器 ---
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=transformer_d_model, 
            nhead=transformer_nhead, 
            dim_feedforward=transformer_d_model * 4,
            dropout=0.3,
            batch_first=True # 使用 (batch, seq, feature) 格式
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=transformer_num_layers)
        
        # --- 4. 全连接输出层 ---
        self.output_layer = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(transformer_d_model // 2, output_bins)
        )
        
    def forward(self, x):
        """
        定义模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length)。

        返回:
            torch.Tensor: 模型的输出，形状为 (batch_size, output_bins)。
        """
        # 1. 调整输入形状以适应Conv1d: (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # 2. 通过CNN提取局部特征: (batch_size, cnn_out_channels, cnn_output_length)
        x = self.cnn_extractor(x)
        
        # 3. 调整形状以适应Transformer: (batch_size, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # 4. 添加位置编码
        # Transformer默认输入是(seq_len, batch, features)，但我们已设为batch_first
        # PositionalEncoding默认是(seq_len, batch, features), 需要调整
        # 然而，直接在batch_first的张量上添加广播的位置编码更简单
        # x: (batch_size, seq_len, d_model)
        # self.pos_encoder.pe: (max_len, 1, d_model), 我们需要(1, seq_len, d_model)
        seq_len = x.size(1)
        pos_encoding = self.pos_encoder.pe[:seq_len, :].permute(1, 0, 2)
        x = x + pos_encoding # 广播加法
        
        # 5. 通过Transformer编码器
        x = self.transformer_encoder(x)
        
        # 6. 全局平均池化 (Global Average Pooling)
        # 取序列维度的平均值，将 (batch, seq_len, features) -> (batch, features)
        x = x.mean(dim=1)
        
        # 7. 通过全连接层得到最终输出
        output = self.output_layer(x)
        
        return output

if __name__ == '__main__':
    # --- 示例：如何使用HybridCNNTransformer模型 ---

    # 1. 定义模型参数
    BATCH_SIZE = 8
    SEQUENCE_LENGTH = 50000   # 示例序列长度
    NUM_BINS = 20             # 示例输出直方图的区间数
    CNN_CHANNELS = 64         # CNN输出通道数
    TRANSFORMER_D_MODEL = 64  # Transformer特征维度
    
    # 2. 实例化模型
    print("--- 实例化CNN-Transformer混合模型 ---")
    model = HybridCNNTransformer(
        input_length=SEQUENCE_LENGTH, 
        output_bins=NUM_BINS,
        cnn_out_channels=CNN_CHANNELS,
        transformer_d_model=TRANSFORMER_D_MODEL,
        transformer_nhead=8, # 注意力头数
        transformer_num_layers=4 # Transformer层数
    )
    print(model)

    # 3. 创建一个虚拟的输入张量
    print(f"\n--- 创建一个虚拟输入张量 ---")
    dummy_input = torch.randint(0, 2, (BATCH_SIZE, SEQUENCE_LENGTH)).float()
    print(f"输入张量形状: {dummy_input.shape}")

    # 4. 执行前向传播
    print("\n--- 执行前向传播 ---")
    output = model(dummy_input)
    print(f"输出张量形状: {output.shape}")
    
    # 5. 检查输出
    print("\n--- 检查softmax后的输出 ---")
    probabilities = F.softmax(output, dim=1)
    print("单个样本的输出概率（示例）:")
    print(probabilities[0])
    print(f"概率和: {torch.sum(probabilities[0])}")

    # 6. 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params:,}") 