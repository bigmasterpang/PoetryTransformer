# -*- coding: utf-8 -*-
"""
全量古诗处理脚本（兼容paragraphs/content字段）
生成格式：前半句-后半句
"""
import os
import json
import re
from opencc import OpenCC

def clean_text(text):
    """文本清理：移除注释/标点/空格"""
    text = re.sub(r'（[^）]*）|\([^)]*\)', '', text)  # 移除注释
    text = re.sub(r'[，。！？、：；]', ' ', text)      # 替换标点为空格
    return re.sub(r'\s+', ' ', text).strip()         # 合并空格

def get_poem_lines(poem):
    """智能获取诗句行：优先content，其次paragraphs"""
    if 'content' in poem:
        return poem['content']
    elif 'paragraphs' in poem:
        return poem['paragraphs']
    return []

def process_poetry_collection(input_dir, output_file):
    """处理单个诗集文件夹"""
    converter = OpenCC('t2s')
    total_pairs = 0
    
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if not file_name.endswith('.json'):
                continue
                
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if not isinstance(data, list):
                        continue
                        
                    for poem in data:
                        lines = get_poem_lines(poem)
                        if not lines:
                            continue
                            
                        for line in lines:
                            clean_line = clean_text(converter.convert(line))
                            parts = [p for p in clean_line.split(' ') if p]
                            # 生成对子（两两一组）
                            for i in range(0, len(parts)-1, 2):
                                if i+1 < len(parts):
                                    output_file.write(f"{parts[i]}，{parts[i+1]}。\n")
                                    total_pairs += 1
                                
                except Exception as e:
                    print(f"处理文件出错 {file_name}: {str(e)}")
    
    return total_pairs

def main():
    base_dir = 'chinese-poetry-master/'
    output_path = 'all_poetry_pairs.txt'
    
    # 需要处理的诗集文件夹列表
    collections = [
        '曹操诗集',
        '全唐诗',
        '水墨唐诗',
        '宋词',
        '楚辞',
        '诗经'
        # 可按需添加其他诗集
    ]
    
    total_pairs = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for collection in collections:
            collection_dir = os.path.join(base_dir, collection)
            if not os.path.exists(collection_dir):
                print(f"警告：目录不存在 {collection_dir}")
                continue
            
            print(f"正在处理 {collection}...")
            count = process_poetry_collection(collection_dir, f_out)
            total_pairs += count
            print(f"  └─ 生成 {count} 个对子")
    
    print(f"\n处理完成，共生成 {total_pairs} 个纯净对子")
    print(f"结果已保存到 {output_path}")
    print("\n示例输出：")
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in list(f.readlines())[:10]:  # 显示前10个示例
            print(line.strip())

if __name__ == "__main__":
    main()
