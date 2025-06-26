import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridCNNBiLSTM(nn.Module):
    """
    一个混合模型，结合了卷积神经网络（CNN）和双向长短期记忆网络（BiLSTM）。
    该模型旨在从基因组序列数据中推断人口统计学历史。

    模型结构：
    1.  一个一维卷积层（CNN）用于从输入序列中提取局部特征模式。
    2.  一个双向LSTM层（BiLSTM）用于捕捉这些局部特征之间的长期依赖关系。
    3.  一个全连接层（FC）用于将BiLSTM的输出映射到最终的预测结果（例如，归一化的TMRCA直方图）。
    """
    def __init__(self, input_length, output_bins, cnn_out_channels=32, kernel_size=11, lstm_hidden_size=64, lstm_num_layers=2):
        """
        初始化混合模型。

        参数:
            input_length (int): 输入序列的长度。
            output_bins (int): 输出的维度（例如，直方图的区间数）。
            cnn_out_channels (int): CNN层的输出通道数。
            kernel_size (int): CNN卷积核的大小。
            lstm_hidden_size (int): LSTM层的隐藏状态维度。
            lstm_num_layers (int): LSTM层的层数。
        """
        super(HybridCNNBiLSTM, self).__init__()
        
        # --- CNN层 ---
        # 输入形状: (batch_size, 1, input_length)
        self.cnn_layer = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(0.3)
        )
        
        # 计算CNN输出后的序列长度
        cnn_output_length = input_length // 4
        
        # --- BiLSTM层 ---
        self.lstm_layer = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            bidirectional=True,
            batch_first=True, # 输入和输出张量以 (batch, seq, feature) 格式提供
            dropout=0.3 if lstm_num_layers > 1 else 0
        )
        
        # --- 全连接层 ---
        # BiLSTM的输出是前向和后向隐藏状态的拼接，所以维度是 2 * lstm_hidden_size
        self.fc_layer = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(lstm_hidden_size, output_bins)
        )

    def forward(self, x):
        """
        定义模型的前向传播。

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, sequence_length)。

        返回:
            torch.Tensor: 模型的输出，形状为 (batch_size, output_bins)。
        """
        # 1. 输入需要增加一个通道维度以适用于Conv1d
        # 形状: (batch_size, 1, sequence_length)
        x = x.unsqueeze(1)
        
        # 2. 通过CNN层提取局部特征
        # 形状: (batch_size, cnn_out_channels, cnn_output_length)
        cnn_output = self.cnn_layer(x)
        
        # 3. 调整张量维度以适应LSTM输入
        # LSTM需要 (batch_size, sequence_length, features)
        # 形状: (batch_size, cnn_output_length, cnn_out_channels)
        lstm_input = cnn_output.permute(0, 2, 1)
        
        # 4. 通过BiLSTM层捕捉长期依赖
        # lstm_out 形状: (batch_size, seq_len, num_directions * hidden_size)
        # h_n 形状: (num_layers * num_directions, batch_size, hidden_size)
        # c_n 形状: (num_layers * num_directions, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm_layer(lstm_input)
        
        # 5. 我们使用最后一个时间步的隐藏状态作为全连接层的输入
        # h_n[-2,:,:] 是最后一层的前向LSTM的隐藏状态
        # h_n[-1,:,:] 是最后一层的后向LSTM的隐藏状态
        # 将它们拼接起来
        # 形状: (batch_size, 2 * lstm_hidden_size)
        last_hidden_state = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        
        # 6. 通过全连接层得到最终输出
        # 形状: (batch_size, output_bins)
        output = self.fc_layer(last_hidden_state)
        
        # 返回原始输出（logits），损失函数（如CrossEntropyLoss）通常会内置softmax
        return output

if __name__ == '__main__':
    # --- 示例：如何使用HybridCNNBiLSTM模型 ---

    # 1. 定义模型参数
    BATCH_SIZE = 16
    SEQUENCE_LENGTH = 100000  # 示例序列长度
    NUM_BINS = 20             # 示例输出直方图的区间数

    # 2. 实例化模型
    print("--- 实例化混合模型 ---")
    model = HybridCNNBiLSTM(input_length=SEQUENCE_LENGTH, output_bins=NUM_BINS)
    print(model)

    # 3. 创建一个虚拟的输入张量
    # 模拟一个批次的二进制序列数据
    print(f"\n--- 创建一个虚拟输入张量 ---")
    dummy_input = torch.randint(0, 2, (BATCH_SIZE, SEQUENCE_LENGTH)).float()
    print(f"输入张量形状: {dummy_input.shape}")

    # 4. 执行前向传播
    print("\n--- 执行前向传播 ---")
    output = model(dummy_input)
    print(f"输出张量形状: {output.shape}")
    
    # 5. 检查输出（应用softmax后）
    print("\n--- 检查softmax后的输出 ---")
    probabilities = F.softmax(output, dim=1)
    print("单个样本的输出概率（示例）:")
    print(probabilities[0])
    print(f"概率和: {torch.sum(probabilities[0])}")

    # 6. 打印模型参数数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型总参数量: {total_params:,}")
