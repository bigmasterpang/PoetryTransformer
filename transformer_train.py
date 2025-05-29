# -*- coding: utf-8 -*-
"""
自回归古诗生成Transformer模型
数据格式：上联，下联。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import time
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from collections import Counter
import random

# 配置参数
class Config:
    def __init__(self):
        self.batch_size = 256
        self.epochs = 300
        self.lr = 0.0001
        self.embed_size = 256
        self.num_layers = 4
        self.heads = 8
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = 20  # 最大生成长度
        self.vocab_size = 5000  # 初始值，实际根据数据调整
        self.save_path = "poetry_generator.pth"
        self.test_inputs = ["春风又绿江南岸", "床前明月光"]  # 多个测试输入
        self.temperature = 0.9  # 生成温度
        self.top_k = 30        # Top-k采样
        self.pad_idx = 0       # 填充标记索引
        self.sos_idx = 1       # 开始标记索引
        self.eos_idx = 2       # 结束标记索引
        self.checkpoint_dir = "Semantic"  # 新增：检查点目录
        self.save_interval = 2  # 每2个epoch保存一次
        self.resume = True  # 是否从检查点恢复
        self.max_samples = 50000  # 新增：最大加载样本数
        self.semantic_dropout = 0.2  # 语义注意力层的dropout
        self.early_stop_patience = 3   # 验证损失不下降的epoch数阈值
        self.min_delta = 0.001         # 视为改进的最小变化量
        
        # 创建检查点目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)

# 自定义数据集类
class PoetryDataset(Dataset):
    def __init__(self, file_path, config):
        self.config = config
        self.pairs = []  # 存储(上联，下联)对
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
        
        # 加载数据并构建词汇表
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num >= config.max_samples:
                    break
                line = line.strip()
                if line and '，' in line and '。' in line:
                    # 分割上下联
                    parts = line.split('，')
                    if len(parts) == 2:
                        upper = parts[0].strip()
                        lower = parts[1].replace('。', '').strip()
                        if upper and lower:  # 确保非空
                            self.pairs.append((upper, lower))
                            
                            # 更新词汇表（合并上下联词汇）
                            for word in upper + lower:
                                if word not in self.word2idx:
                                    idx = len(self.word2idx)
                                    self.word2idx[word] = idx
                                    self.idx2word[idx] = word
        
        self.config.vocab_size = len(self.word2idx)
        self._analyze_data()
    
    def _analyze_data(self):
        """数据分析"""
        upper_lengths = [len(p[0]) for p in self.pairs]
        lower_lengths = [len(p[1]) for p in self.pairs]
        print(f"\n数据集统计:")
        print(f"总对联数: {len(self.pairs)}")
        print(f"上联平均长度: {np.mean(upper_lengths):.1f} ± {np.std(upper_lengths):.1f}")
        print(f"下联平均长度: {np.mean(lower_lengths):.1f} ± {np.std(lower_lengths):.1f}")
        print(f"最长上联: {max(upper_lengths)}, 最长下联: {max(lower_lengths)}")
        print(f"词汇表大小: {len(self.word2idx)}")
        
        # 显示高频词
        word_counts = Counter()
        for upper, lower in self.pairs:
            word_counts.update(upper)
            word_counts.update(lower)
        print("\n高频词TOP10:", word_counts.most_common(10))
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        upper, lower = self.pairs[idx]
        
        # 编码上联（输入）
        src_indices = [self.word2idx["<SOS>"]] + [self.word2idx[word] for word in upper] + [self.word2idx["<EOS>"]]
        
        # 编码下联（目标）
        tgt_indices = [self.word2idx["<SOS>"]] + [self.word2idx[word] for word in lower] + [self.word2idx["<EOS>"]]
        
        return torch.tensor(src_indices), torch.tensor(tgt_indices)

# 数据整理函数
def collate_fn(batch, config):
    src_batch, tgt_batch = zip(*batch)
    
    # 计算实际最大长度（不包括自动填充的<PAD>）
    src_max_len = max(len(src) for src in src_batch)
    tgt_max_len = max(len(tgt) for tgt in tgt_batch)
    
    # 初始化填充矩阵
    padded_src = torch.full((len(batch), src_max_len), config.pad_idx, dtype=torch.long)
    padded_tgt = torch.full((len(batch), tgt_max_len), config.pad_idx, dtype=torch.long)
    
    for i, (src, tgt) in enumerate(zip(src_batch, tgt_batch)):
        # 上联处理 - 只拷贝实际内容
        padded_src[i, :len(src)] = src[:src_max_len]
        
        # 下联处理 - 只拷贝实际内容
        padded_tgt[i, :len(tgt)] = tgt[:tgt_max_len]
        
        # 不再手动添加<EOS>，因为数据中已包含
    
    return padded_src.to(config.device), padded_tgt.to(config.device)

# Transformer模型
class PoetryGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attention_penalty = 0.1
        
        # 词嵌入层
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        
        self.semantic_attn = SemanticAttention(config.embed_size)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(config.embed_size, config.max_length)
        
        # Transformer结构
        self.transformer = nn.Transformer(
            d_model=config.embed_size,
            nhead=config.heads,
            num_encoder_layers=config.num_layers,
            num_decoder_layers=config.num_layers,
            dim_feedforward=config.embed_size*4,
            dropout=0.1,
            activation='gelu'
        )
        
        # 输出层
        self.fc_out = nn.Linear(config.embed_size, config.vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, 
            src_key_padding_mask=None, tgt_key_padding_mask=None):
        # 嵌入层
        src_embed = self.embedding(src) * math.sqrt(self.config.embed_size)
        tgt_embed = self.embedding(tgt) * math.sqrt(self.config.embed_size)
        
        src_embed = src_embed + self.semantic_attn(src_embed)
        tgt_embed = tgt_embed + self.semantic_attn(tgt_embed)
        
        # 位置编码
        src_embed = self.pos_encoder(src_embed)
        tgt_embed = self.pos_encoder(tgt_embed)
        
        # 调整维度 [batch, seq, features] -> [seq, batch, features]
        src_embed = src_embed.permute(1, 0, 2)
        tgt_embed = tgt_embed.permute(1, 0, 2)
        
        # 生成目标掩码
        tgt_mask = self.transformer.generate_square_subsequent_mask(
            tgt.size(1)
        ).to(self.config.device)
        
        # Transformer计算
        output = self.transformer(
            src_embed, 
            tgt_embed, 
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # 输出层
        output = self.fc_out(output.permute(1, 0, 2))
        return output

# 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(1), :]

class SemanticAttention(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_weights = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(x.size(-1)))
        return torch.matmul(attn_weights, V)

# 训练函数
def train(model, dataloader, optimizer, criterion, epoch, config, train_losses, dataset):
    model.train()
    total_loss = 0
    batch_losses = []
    
    debug_samples = 3
    printed = False
    
    teacher_forcing_ratio = max(0.7, 0.9 - epoch*0.01)
    
    for batch_idx, (src, tgt) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        src, tgt = src.to(config.device), tgt.to(config.device)
        
        # 打印前3个batch的输入输出
        # if not printed and batch_idx < debug_samples:
        #     print(f"\n=== 调试样本 {batch_idx+1} ===")
        #     print("输入序列 (src):")
        #     for i in range(min(3, src.size(0))):  # 每个batch打印前3个样本
        #         src_tokens = src[i].cpu().numpy()
        #         src_text = ''.join([dataset.idx2word[idx] for idx in src_tokens if idx != config.pad_idx])
        #         print(f"样本 {i+1}: {src_tokens} -> {src_text}")
            
        #     print("\n目标序列 (tgt):")
        #     for i in range(min(3, tgt.size(0))):
        #         tgt_tokens = tgt[i].cpu().numpy()
        #         tgt_text = ''.join([dataset.idx2word[idx] for idx in tgt_tokens if idx != config.pad_idx])
        #         print(f"样本 {i+1}: {tgt_tokens} -> {tgt_text}")
            
        #     if batch_idx == debug_samples-1:
        #         printed = True
        #         print("\n" + "="*50 + "\n")
        
        # 准备输入和目标（去掉最后一个token作为decoder输入）
        tgt_input = tgt[:, :-1]  # 形状: [batch_size, seq_len-1]
        tgt_output = tgt[:, 1:]  # 形状: [batch_size, seq_len-1]
        
        # 生成所有必要的掩码
        src_padding_mask = (src == config.pad_idx).to(config.device)  # 源序列填充掩码
        tgt_padding_mask = (tgt_input == config.pad_idx).to(config.device)  # 目标序列填充掩码
        tgt_mask = model.transformer.generate_square_subsequent_mask(
            tgt_input.size(1)  # 目标序列长度
        ).to(config.device)  # 自回归掩码
        
        # 动态teacher forcing
        use_teacher_forcing = random.random() < teacher_forcing_ratio
        if use_teacher_forcing:
            output = model(src, tgt_input, 
                          src_key_padding_mask=src_padding_mask,
                          tgt_key_padding_mask=tgt_padding_mask,
                          tgt_mask=tgt_mask)
        else:
            # 自回归训练
            output = []
            for i in range(tgt_input.size(1)):
                out = model(src, tgt_input[:, :i+1],
                           src_key_padding_mask=src_padding_mask,
                           tgt_key_padding_mask=tgt_padding_mask[:, :i+1],
                           tgt_mask=tgt_mask[:i+1, :i+1])
                output.append(out[:, -1, :])
            output = torch.stack(output, dim=1)
            
        # if not printed and batch_idx < debug_samples:
        #     print("\n模型输出 (output):")
        #     print(f"输出形状: {output.shape}")  # [batch_size, seq_len-1, vocab_size]
            
        #     # 打印第一个样本的前5个预测token
        #     sample_output = output[0].argmax(-1).cpu().numpy()
        #     print(f"第一个样本的预测token: {sample_output[:5]}")
            
        #     # 解码预测结果
        #     pred_text = ''.join([dataset.idx2word[idx] for idx in sample_output if idx not in [config.pad_idx, config.sos_idx, config.eos_idx]])
        #     print(f"解码预测: {pred_text[:20]}...")  # 只显示前20个字符
            
        #     # 打印真实目标
        #     true_tokens = tgt_output[0].cpu().numpy()
        #     true_text = ''.join([dataset.idx2word[idx] for idx in true_tokens if idx not in [config.pad_idx, config.sos_idx, config.eos_idx]])
        #     print(f"真实目标: {true_text[:20]}...")
        
        # 新增重复词惩罚
        repeated_mask = (tgt_output[:, :-1] == tgt_output[:, 1:]).float()
        probs = torch.softmax(output, dim=-1)  # 转换为概率
        
        # 惩罚重复词的高概率
        penalty = 0.3 * (probs[:, :-1] * repeated_mask.unsqueeze(-1)).mean()
        
        loss = criterion(output.reshape(-1, config.vocab_size), 
                        tgt_output.reshape(-1)) + penalty
         
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        batch_losses.append(loss.item())
    
    epoch_loss = sum(batch_losses) / len(batch_losses)
    train_losses.append(epoch_loss)
    return train_losses

# 生成函数
def generate(model, start_words, dataset, config):
    model.eval()
    results = []
    
    for input_str in start_words:
        # 编码输入
        input_idx = [dataset.word2idx["<SOS>"]] + [
            dataset.word2idx[w] for w in input_str 
            if w in dataset.word2idx
        ]
        src = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(config.device)
        
        # 初始化目标序列（与训练时一致）
        tgt = torch.tensor([[dataset.word2idx["<SOS>"]]], dtype=torch.long).to(config.device)
        
        # 生成源序列padding mask
        src_padding_mask = (src == dataset.word2idx["<PAD>"]).to(config.device)
        
        with torch.no_grad():
            for i in range(config.max_length):
                tgt_mask = model.transformer.generate_square_subsequent_mask(
                    tgt.size(1)
                ).to(config.device)
                
                output = model(
                    src, tgt,
                    tgt_mask=tgt_mask,
                    src_key_padding_mask=src_padding_mask
                )
                
                # 改用概率采样而非argmax
                next_token = output[:, -1, :] / config.temperature
                probs = torch.softmax(next_token, dim=-1)
                
                # 更宽松的终止条件
                if i > 5:  # 至少生成5个字
                    eos_prob = probs[:, dataset.word2idx["<EOS>"]].item()
                    if eos_prob > 0.5:  # EOS概率超过50%才停止
                        break
                
                # Top-k采样
                if config.top_k > 0:
                    top_probs, top_indices = torch.topk(probs, config.top_k)
                    probs = torch.zeros_like(probs).scatter_(-1, top_indices, top_probs)
                
                next_word = torch.multinomial(probs, num_samples=1)
                tgt = torch.cat([tgt, next_word], dim=1)
        
        # 解码时保留更多内容
        output_words = ''.join([
            dataset.idx2word[idx.item()] 
            for idx in tgt[0][1:]  # 保留直到EOS
            if idx.item() not in [dataset.word2idx["<PAD>"], dataset.word2idx["<EOS>"]]
        ])
        results.append(f"输入: '{input_str}' → 生成: '{output_words}'")
    
    return results

def save_checkpoint(epoch, model, optimizer, train_losses, val_losses, config, dataset):
    """保存完整训练状态"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            **vars(config),
            'word2idx': dataset.word2idx,  # 新增
            'idx2word': dataset.idx2word,  # 新增
            'vocab_size': len(dataset.word2idx)  # 动态词汇表大小
        }
    }
    
    # 带时间戳的检查点
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(
        config.checkpoint_dir, 
        f"epoch_{epoch}_{timestamp}.pth"
    )
    torch.save(checkpoint, path)
    
    # 同时保存最新版本
    torch.save(checkpoint, os.path.join(config.checkpoint_dir, "latest.pth"))
    print(f"\n检查点已保存至: {path}")
    
def load_checkpoint(config, model, optimizer=None):
    """加载最新检查点"""
    latest_path = os.path.join(config.checkpoint_dir, "latest.pth")
    if not os.path.exists(latest_path):
        return None
    
    checkpoint = torch.load(latest_path)
    
    # 恢复模型和优化器
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"\n从检查点恢复: epoch={checkpoint['epoch']}, loss={checkpoint['train_losses'][-1]:.4f}")
    return checkpoint

def evaluate(model, dataloader, criterion, config):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for src, tgt in tqdm(dataloader, desc="验证中"):
            src, tgt = src.to(config.device), tgt.to(config.device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input)
            loss = criterion(output.reshape(-1, config.vocab_size), tgt_output.reshape(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, 'b-', label='Train Loss')
    plt.plot(val_losses, 'r-', label='Validation Loss')
    
    # 标记最佳验证点
    best_idx = np.argmin(val_losses)
    plt.scatter(best_idx, val_losses[best_idx], c='red', s=100, 
               label=f'Best Val Loss: {val_losses[best_idx]:.4f}')
    
    plt.title('Training Progress with Early Stopping')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve_with_early_stop.png')
    plt.show()

# 主函数
def main():
    config = Config()
    
    # 加载数据集
    dataset = PoetryDataset("all_poetry_pairs_processed.txt", config)
    
    # 创建数据加载器
    dataloader = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, config)
    )
    
    # 初始化模型
    model = PoetryGenerator(config).to(config.device)
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=config.pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    # 断点续训（如果启用且存在检查点）
    if config.resume:
        checkpoint = load_checkpoint(config, model, optimizer)
        if checkpoint:
            start_epoch = checkpoint['epoch'] + 1
            train_losses = checkpoint['train_losses']
            val_losses = checkpoint['val_losses']
            best_val_loss = min(checkpoint['val_losses'])
            early_stop_counter = 0  # 重置计数器
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    early_stop_counter = 0
    stop_training = False
    start_epoch = 1
    for epoch in range(start_epoch, config.epochs + 1):
        if stop_training:
            break
        # 训练和验证
        train_losses = train(model, dataloader, optimizer, criterion, epoch, config, train_losses, dataset)
        val_loss = evaluate(model, dataloader, criterion, config)
        val_losses.append(val_loss)
        
        if val_loss < best_val_loss - config.min_delta:
            best_val_loss = val_loss
            early_stop_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), f"best_{config.save_path}")
        else:
            early_stop_counter += 1
            if early_stop_counter >= config.early_stop_patience:
                print(f"\n早停触发！验证损失连续{early_stop_counter}个epoch未下降")
                stop_training = True
        
        # 定期输出和保存
        if epoch % config.save_interval == 0 or epoch == config.epochs:
            print(f"Epoch {epoch}: Train Loss={train_losses[-1]:.4f}, Val Loss={val_loss:.4f}, "
                  f"Patience={early_stop_counter}/{config.early_stop_patience}")
            
            # 生成示例
            results = generate(model, config.test_inputs, dataset, config)
            for res in results:
                print(res)
            
            # 保存检查点
            save_checkpoint(epoch, model, optimizer, train_losses, val_losses, config, dataset)
    
    # 最终保存和绘图
    torch.save(model.state_dict(), config.save_path)
    plot_losses(train_losses, val_losses)

if __name__ == "__main__":
    main()
