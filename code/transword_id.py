import os
import sys
import pickle
"""
将词汇表vocab.txt转换为word2id和id2word字典
"""
def build_vocab(vocab_path, min_count=1):
    word2id = {}
    id2word = {}
    
    # 手动定义特殊token，放在最前面，固定索引
    special_tokens = ['<pad>', '<unk>']
    for idx, token in enumerate(special_tokens):
        word2id[token] = idx
        id2word[idx] = token
    
    current_id = len(special_tokens)  # 普通词从索引2开始
    
    # 严格按vocab.txt的行顺序（词频降序）分配索引
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            word, count = line.strip().split()
            count = int(count)
            if count < min_count:
                break  # 词表已降序，遇到第一个低于阈值的词就终止
            word2id[word] = current_id
            id2word[current_id] = word
            current_id += 1
    return word2id, id2word


if __name__ == '__main__':
    vocab_path = r'D:\研一下\python_project\TextCNN\Data\cnews_corpus\vocab.txt'
    word2id, id2word = build_vocab(vocab_path)
    print(word2id)
    print(id2word)
    # 打包成一个字典，统一保存
    vocab_dict = {
        "word2id": word2id,
        "id2word": id2word,
        "vocab_size": len(word2id),
        "min_count": 1  # 顺便保存参数，方便后续核对
    }

    # 保存到文件
    pkl_path = r'D:\研一下\python_project\TextCNN\Data\cnews_corpus\vocab.pkl'
    with open(pkl_path, "wb") as f:
        pickle.dump(vocab_dict, f)
    print(f"词汇表pkl文件已保存到 {pkl_path}")


