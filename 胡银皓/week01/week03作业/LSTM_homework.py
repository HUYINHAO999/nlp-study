import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

# 1. 数据准备
file_path = 'jaychou_lyrics.txt'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]
    text = "".join(lines)  # 用空字符串连接，保持原始结构

# 找出所有的独立字符并创建映射
vocab = sorted(list(set(text)))
char_to_idx = {char: idx for idx, char in enumerate(vocab)}
idx_to_char = {idx: char for idx, char in enumerate(vocab)}
vocab_size = len(vocab)

# 转换为整数序列
text_as_int = np.array([char_to_idx[c] for c in text])


# 2. 定义数据集和数据加载器
class LyricsDataset(Dataset):
    def __init__(self, text, seq_length):
        self.text = text
        self.seq_length = seq_length
        self.data_size = len(text) - seq_length

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        input_seq = self.text[idx:idx + self.seq_length]
        target_seq = self.text[idx + 1:idx + self.seq_length + 1]
        return (torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long))


# 创建数据集
seq_length = 100
dataset = LyricsDataset(text_as_int, seq_length)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 3. 定义三种模型：RNN、LSTM、GRU
class BaseRNN(nn.Module):
    def __init__(self, model_type: str, vocab_size: int, embedding_dim: int, hidden_dim: int):
        """
        Args:
            model_type: 'rnn', 'lstm', or 'gru'
        """
        super(BaseRNN, self).__init__()
        self.model_type = model_type
        self.hidden_dim = hidden_dim

        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # 循环神经网络层
        if model_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif model_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # 全连接层
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # 嵌入层
        embedded = self.embedding(x)  # [batch, seq, embedding]

        # RNN层
        if hidden is None:
            output, hidden = self.rnn(embedded)
        else:
            output, hidden = self.rnn(embedded, hidden)

        # 重塑并全连接
        batch_size, seq_len = x.size(0), x.size(1)
        output = output.reshape(-1, self.hidden_dim)  # [batch*seq, hidden]
        logits = self.fc(output)  # [batch*seq, vocab_size]

        return logits, hidden

    def init_hidden(self, batch_size: int):
        """初始化隐藏状态"""
        if self.model_type == 'lstm':
            # LSTM需要(hidden_state, cell_state)
            return (torch.zeros(1, batch_size, self.hidden_dim),
                    torch.zeros(1, batch_size, self.hidden_dim))
        else:
            # RNN和GRU只需要hidden_state
            return torch.zeros(1, batch_size, self.hidden_dim)


# 4. 训练函数
def train_model(model_type: str, num_epochs: int = 2, model_save_path: str = None):
    """训练指定类型的模型"""
    print(f"\n{'=' * 50}")
    print(f"训练 {model_type.upper()} 模型")
    print('=' * 50)

    # 设置参数
    embedding_dim = 32
    hidden_dim = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建模型
    model = BaseRNN(model_type, vocab_size, embedding_dim, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 尝试加载已有模型
    if model_save_path and os.path.exists(model_save_path):
        print(f"加载已有模型: {model_save_path}")
        model.load_state_dict(torch.load(model_save_path, map_location=device))
        return model, []

    # 训练
    losses = []
    start_time = time.time()

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # 初始化隐藏状态
            if model_type == 'lstm':
                hidden = model.init_hidden(batch_size)
                hidden = (hidden[0].to(device), hidden[1].to(device))
                hidden = (hidden[0].detach(), hidden[1].detach())  # 分离梯度
            else:
                hidden = model.init_hidden(batch_size)
                hidden = hidden.to(device)
                hidden = hidden.detach()  # 分离梯度

            # 前向传播
            model.zero_grad()
            logits, hidden = model(inputs, hidden)
            loss = criterion(logits, targets.view(-1))

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            epoch_loss += loss.item()

            if (i + 1) % 100 == 0:
                avg_loss = epoch_loss / (i + 1)
                losses.append(avg_loss)
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], '
                      f'Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}')

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch + 1} 完成，平均损失: {avg_epoch_loss:.4f}')

    training_time = time.time() - start_time
    print(f"训练完成，总时间: {training_time:.2f}秒")

    # 保存模型
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
        print(f"模型已保存至 {model_save_path}")

    return model, losses


# 5. 修复后的歌词生成函数
def generate_text(model, start_string: str, num_generate: int = 100,
                  temperature: float = 1.0, model_type: str = 'lstm'):
    """生成文本，支持温度采样"""
    model.eval()
    device = next(model.parameters()).device

    # 将起始字符串转换为张量
    input_seq = [char_to_idx[s] for s in start_string]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    generated_text = start_string

    # 初始化隐藏状态
    if model_type == 'lstm':
        hidden = model.init_hidden(1)
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = model.init_hidden(1)
        hidden = hidden.to(device)

    with torch.no_grad():
        for _ in range(num_generate):
            # 前向传播
            logits, hidden = model(input_tensor, hidden)

            # 关键修复：获取最后一个时间步的logits
            # logits形状: [batch*seq_len, vocab_size]
            # 我们需要最后一个字符的预测
            last_logits = logits[-1:]  # 取最后一行，形状: [1, vocab_size]

            # 应用温度参数
            if temperature != 1.0:
                last_logits = last_logits / temperature

            # 使用softmax获取概率并采样
            probs = torch.softmax(last_logits, dim=1)  # [1, vocab_size]
            predicted_id = torch.multinomial(probs, num_samples=1).item()

            # 添加到生成文本
            generated_text += idx_to_char[predicted_id]

            # 更新输入（只保留最后一个预测字符）
            input_tensor = torch.tensor([[predicted_id]],
                                        dtype=torch.long).to(device)

    return generated_text


# 6. 评估函数
def evaluate_model(model, dataloader, model_type: str, num_batches: int = 10):
    """评估模型在验证集上的性能"""
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if i >= num_batches:
                break

            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # 初始化隐藏状态
            if model_type == 'lstm':
                hidden = model.init_hidden(batch_size)
                hidden = (hidden[0].to(device), hidden[1].to(device))
            else:
                hidden = model.init_hidden(batch_size)
                hidden = hidden.to(device)

            # 前向传播
            logits, _ = model(inputs, hidden)

            # 计算损失
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits, targets.view(-1))
            total_loss += loss.item()

            # 计算准确率
            predictions = torch.argmax(logits, dim=1)
            correct = (predictions == targets.view(-1)).sum().item()
            total_correct += correct
            total_samples += targets.numel()

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples * 100

    return avg_loss, accuracy


# 7. 简单文本生成函数（用于快速测试）
def quick_generate(model, model_type: str, prompt: str = "爱情", length: int = 50):
    """快速生成文本用于测试"""
    model.eval()
    device = next(model.parameters()).device

    # 初始化
    input_seq = [char_to_idx[ch] for ch in prompt]
    input_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)

    if model_type == 'lstm':
        hidden = model.init_hidden(1)
        hidden = (hidden[0].to(device), hidden[1].to(device))
    else:
        hidden = model.init_hidden(1)
        hidden = hidden.to(device)

    generated = prompt

    with torch.no_grad():
        for _ in range(length):
            logits, hidden = model(input_tensor, hidden)

            # 取最后一个时间步的预测
            last_logits = logits[-1:]  # [1, vocab_size]

            # 贪心选择
            predicted_id = torch.argmax(last_logits, dim=1).item()

            # 添加到生成文本
            generated += idx_to_char[predicted_id]

            # 更新输入
            input_tensor = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

    return generated


# 8. 主函数：对比三种模型
def main():
    # 训练三种模型
    model_types = ['rnn', 'lstm', 'gru']
    model_paths = {
        'rnn': 'rnn_model.pth',
        'lstm': 'lstm_model.pth',
        'gru': 'gru_model.pth'
    }

    models = {}
    all_losses = {}
    results = {}

    # 为了快速演示，只训练一个epoch
    train_epochs = 20  # 可以调整为更大的值以获得更好的效果

    # 训练并评估每个模型
    for model_type in model_types:
        print(f"\n{'=' * 60}")
        print(f"处理 {model_type.upper()} 模型")
        print('=' * 60)

        # 训练模型
        model, losses = train_model(
            model_type=model_type,
            num_epochs=train_epochs,
            model_save_path=model_paths[model_type]
        )

        models[model_type] = model
        all_losses[model_type] = losses

        # 评估模型
        print("评估模型性能...")
        avg_loss, accuracy = evaluate_model(
            model, dataloader, model_type, num_batches=5
        )
        results[model_type] = {
            'loss': avg_loss,
            'accuracy': accuracy
        }

        print(f"\n{model_type.upper()} 评估结果:")
        print(f"  平均损失: {avg_loss:.4f}")
        print(f"  准确率: {accuracy:.2f}%")

        # 参数数量统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")

        # 生成示例文本（使用修复后的函数）
        start_prompt = "枫叶"
        print(f"\n生成文本示例 (起始: '{start_prompt}'):")

        try:
            generated_text = generate_text(
                model,
                start_string=start_prompt,
                num_generate=30,
                model_type=model_type
            )
            print(f"  {generated_text}")
        except Exception as e:
            print(f"  生成失败: {e}")
            # 使用简化版本
            simple_text = quick_generate(model, model_type, start_prompt, 20)
            print(f"  简化版本: {simple_text}")

    # 9. 可视化对比结果（如果有损失数据）
    if any(all_losses.values()):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        for model_type in model_types:
            if all_losses[model_type]:
                plt.plot(all_losses[model_type], label=f'{model_type.upper()}')
        plt.xlabel('Training Steps (x100)')
        plt.ylabel('Loss')
        plt.title('Training Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        model_names = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in model_names]

        bars = plt.bar(range(len(model_names)), accuracies, color=['skyblue', 'lightgreen', 'salmon'])
        plt.xlabel('Model Type')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracy Comparison')
        plt.xticks(range(len(model_names)), [m.upper() for m in model_names])

        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom')

        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

    # 10. 打印详细对比结果
    print("\n" + "=" * 60)
    print("模型对比总结")
    print("=" * 60)

    print(f"{'Model':<8} {'Accuracy (%)':<15} {'Loss':<10} {'Params':<12}")
    print("-" * 50)

    for model_type in model_types:
        model = models[model_type]
        total_params = sum(p.numel() for p in model.parameters())

        print(f"{model_type.upper():<8} "
              f"{results[model_type]['accuracy']:<15.2f} "
              f"{results[model_type]['loss']:<10.4f} "
              f"{total_params:<12,}")

    # 11. 更多文本生成示例
    print("\n" + "=" * 60)
    print("更多文本生成示例")
    print("=" * 60)

    prompts = ["爱情", "梦想", "回忆"]

    for prompt in prompts:
        print(f"\n起始词: '{prompt}'")
        for model_type in model_types:
            try:
                generated = generate_text(
                    models[model_type],
                    start_string=prompt,
                    num_generate=20,
                    model_type=model_type
                )
                print(f"{model_type.upper():<6}: {generated}")
            except:
                # 如果失败，使用简化版本
                generated = quick_generate(models[model_type], model_type, prompt, 15)
                print(f"{model_type.upper():<6}: {generated}")


if __name__ == "__main__":
    main()