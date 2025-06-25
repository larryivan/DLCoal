# DLCoal - 深度学习模型用于推断群体历史

这个项目使用深度学习方法从基因组序列中推断群体历史。它使用msprime模拟基因组数据，并训练深度学习模型来预测TMRCA（最近共同祖先时间）分布。

## 项目结构

- `data_generator.py` - 使用msprime生成模拟数据的类
- `main.py` - 训练深度学习模型的主脚本
- `simple.py` - 简单的示例脚本

## 依赖项

本项目需要以下Python包：

```bash
msprime
numpy
tensorflow
scipy
tskit
```

您可以使用以下命令安装依赖项：

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple msprime numpy tensorflow scipy tskit
```

## 使用方法

### 生成模拟数据

您可以使用`data_generator.py`中的`PSMCDataGenerator`类来生成模拟数据：

```python
from data_generator import PSMCDataGenerator

# 创建数据生成器
generator = PSMCDataGenerator(
    sequence_length=100000,  # 序列长度
    time_windows=20,         # TMRCA离散时间窗口数量
    max_time=50000           # 最大时间(以代数计)
)

# 生成一批数据
X, y = generator.generate_batch(batch_size=5)

# 打印结果的形状
print(f"输入序列形状: {X.shape}")
print(f"输出序列形状: {y.shape}")
```

### 训练深度学习模型

运行`main.py`来训练深度学习模型：

```bash
python main.py
```

这将创建一个卷积神经网络模型来预测从PSMC输入序列（0和1）到TMRCA时间窗口分布的映射。

## 数据生成器参数

`PSMCDataGenerator`类接受以下参数：

- `sequence_length` - 模拟序列的长度（默认：1,000,000）
- `sample_size` - 样本大小（默认：2，用于二倍体个体）
- `recombination_rate` - 重组率（默认：1e-8）
- `mutation_rate` - 突变率（默认：1e-8）
- `time_windows` - TMRCA离散时间窗口数量（默认：20）
- `max_time` - 最大时间（以代数计，默认：100,000）

## 输入和输出格式

- **输入**：PSMC格式的序列，表示为0和1的序列，其中1表示杂合位点，0表示同源位点。
- **输出**：离散时间窗口中TMRCA的位点数量，经过softmax归一化。

## 注意事项

- 数据生成器在每次运行时都会生成不同的随机数据。
- 深度学习模型使用卷积神经网络来捕获序列模式。
- 模型使用softmax激活函数输出TMRCA分布。
