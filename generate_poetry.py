# -*- coding: utf-8 -*-
"""
古诗生成模型调用程序
使用方法：python generate_poetry.py
"""

import torch
import argparse
from transformer_train import PoetryGenerator, Config 
import os

def load_model(checkpoint_path, device):
    """正确加载模型的方法"""
    # 先加载配置
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = Config()
    
    # 用检查点中的实际词汇表大小更新配置
    config.vocab_size = checkpoint['config']['vocab_size']  # 应该是6806
    
    # 初始化模型
    model = PoetryGenerator(config).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 恢复词汇表
    dataset_info = {
        'word2idx': checkpoint['config']['word2idx'],
        'idx2word': {int(k):v for k,v in checkpoint['config']['idx2word'].items()}
    }
    
    return model, dataset_info

def generate_single(model, input_str, dataset_info, config):
    """单条生成函数（严格模式，遇到未知字符报错）"""
    # 检查未知字符
    unknown_chars = [
        char for char in input_str 
        if char not in dataset_info['word2idx']
    ]
    
    if unknown_chars:
        raise ValueError(
            f"输入包含{len(unknown_chars)}个未知字符: {''.join(unknown_chars)}\n"
            f"可用字符范围：{len(dataset_info['word2idx'])-4}个汉字+4个特殊标记"
        )
    
    # 编码输入（此时所有字符都已知）
    input_idx = [dataset_info['word2idx']["<SOS>"]] + [
        dataset_info['word2idx'][char] 
        for char in input_str
    ]
    
    src = torch.tensor(input_idx, dtype=torch.long).unsqueeze(0).to(config.device)
    tgt = torch.tensor([[dataset_info['word2idx']["<SOS>"]]], dtype=torch.long).to(config.device)
    
    with torch.no_grad():
        for _ in range(config.max_length):
            tgt_mask = model.transformer.generate_square_subsequent_mask(
                tgt.size(1)
            ).to(config.device)
            
            output = model(src, tgt, tgt_mask=tgt_mask)
            next_token = output[:, -1, :] / config.temperature
            probs = torch.softmax(next_token, dim=-1)
            
            if config.top_k > 0:
                top_k = min(config.top_k, probs.size(-1))
                top_probs, top_indices = torch.topk(probs, top_k)
                probs = torch.zeros_like(probs).scatter_(-1, top_indices, top_probs)
            
            next_word = torch.multinomial(probs, num_samples=1)
            tgt = torch.cat([tgt, next_word], dim=1)
            
            if next_word.item() == dataset_info['word2idx']["<EOS>"]:
                break
    
    # 解码结果
    output_words = ''.join([
        dataset_info['idx2word'][idx.item()] 
        for idx in tgt[0][1:]  # 跳过<SOS>
        if idx.item() not in [
            dataset_info['word2idx']["<PAD>"], 
            dataset_info['word2idx']["<EOS>"]
        ]
    ])
    
    return output_words

def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型和配置
    model, dataset_info = load_model("Semantic/latest.pth", device)
    config = Config()
    
    print("="*50)
    print("古诗生成器已加载（输入q退出）")
    print(f"可用字符数：{len(dataset_info['word2idx'])-4}个汉字+4个特殊标记")
    print("="*50)
    
    while True:
        try:
            # 获取用户输入
            user_input = input("\n请输入上联：").strip()
            
            # 退出条件
            if user_input.lower() in ['q', 'quit', 'exit']:
                print("感谢使用，再见！")
                break
                
            # 空输入处理
            if not user_input:
                print("！输入不能为空")
                continue
                
            # 生成诗句
            output = generate_single(model, user_input, dataset_info, config)
            print(f"生成下联：{output}")
            
            # 显示5个推荐示例
            print("\n其他示例：")
            examples = random.sample([
                "白日依山尽", "红豆生南国", "海上生明月", 
                "春眠不觉晓", "锄禾日当午"
            ], 3)
            for ex in examples:
                print(f"  {ex} → {generate_single(model, ex, dataset_info, config)}")
                
        except ValueError as e:
            print(f"！输入错误：{e}")
            print("提示：请使用常见汉字，避免生僻字和符号")
        except KeyboardInterrupt:
            print("\n检测到中断，程序退出")
            break
        except Exception as e:
            print(f"！发生未知错误：{str(e)}")
            break

if __name__ == "__main__":
    import random  # 添加在文件开头
    main()
