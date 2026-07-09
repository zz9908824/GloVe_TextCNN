#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
预处理 cnews 数据集（分词），使其和 GloVe 格式分词要求一致
1. 去除标签信息（用于 GloVe 词向量训练）
2. 保留标签信息（用于 TextCNN 分类）
3. 中文分词处理
"""

import os
import re
import jieba
from pathlib import Path

def clean_text(text):
    """
    清理文本：移除标点、特殊字符
    """
    # 移除标点符号（保留中文标点）
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)

    # 移除多余空格
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def segment_text(text):
    """
    使用 jieba 分词
    """
    cleaned_text = clean_text(text)
    words = jieba.cut(cleaned_text)
    segmented_text = ' '.join(words)
    return segmented_text

def preprocess_cnews_file(input_file, output_corpus_file):
    """
    预处理 cnews 文件，对文本进行分词处理并保存为新文件
    Args:
        input_file: 输入文件路径（包含标签）
        output_corpus_file: 输出语料文件路径（用于TextCNN分类）
    """
    print(f"处理文件: {input_file}")

    # 读取输入文件
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"总行数: {len(lines)}")

    # 处理每一行
    corpus_lines = []  # 用于TextCNN（无标签）

    for line in lines:
        line = line.strip()
        # 分割标签和文本
        parts = line.split(maxsplit=1)
        if len(parts) < 2:
            continue
        label = parts[0]
        text = parts[1]
        # 分词处理
        segmented_text = segment_text(text)
        # TextCNN分类的 cnews 语料（无标签）
        if segmented_text:
            corpus_lines.append(f"{label} {segmented_text}")

    # 保存语料文件（用于TextCNN）
    with open(output_corpus_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(corpus_lines))
    print(f"保存语料文件: {output_corpus_file}（{len(corpus_lines)}行）")

def preprocess_cnews_dataset(input_dir, output_dir):
    """
    预处理 cnews 数据集

    Args:
        input_dir: 输入目录（未经分词处理的cnews训练集验证集测试集文件所在目录）
        output_dir: 输出目录
    """
    print(f"开始预处理 cnews 数据集...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    # 创建输出目录
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # # 处理训练集
    # train_file = os.path.join(input_dir, 'cnews.train.txt')
    # if os.path.exists(train_file):
    #     preprocess_cnews_file(
    #         train_file,
    #         os.path.join(output_dir, 'cnews_train_corpus.txt')  # TextCNN训练集分好词用于分类的语料
    #     )
    # print(f"\n训练集分词处理完成！")

    # 处理验证集
    val_file = os.path.join(input_dir, 'cnews.val.txt')
    if os.path.exists(val_file):
        preprocess_cnews_file(
            val_file,
            os.path.join(output_dir, 'cnews_val_corpus.txt')  # TextCNN验证集分好词用于分类的语料
        )
    print(f"\n验证集分词处理完成！")

    # # 处理测试集
    # test_file = os.path.join(input_dir, 'cnews.test.txt')
    # if os.path.exists(test_file):
    #     preprocess_cnews_file(
    #         test_file,
    #         os.path.join(output_dir, 'cnews_test_corpus.txt')  # TextCNN测试集分类的语料
    #     )
    # print(f"\n测试集分词处理完成！")

    # merge_corpus_files(output_dir)


if __name__ == '__main__':
    # 设置路径
    input_dir = r'D:\研一下\python_project\TextCNN\Data\cnews_corpus'  # 原始cnews未经分词文件在cnews子目录下 cnew.train.txt cnews.test.txt cnews.val.txt
    output_dir = r'D:\研一下\python_project\TextCNN\Data\cnews_corpus\corpus' # 经过分词和清洗的cnews文件保存路径

    # 执行数据集分词预处理
    preprocess_cnews_dataset(input_dir, output_dir)