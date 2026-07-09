import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pickle
from TextCNN import Model, MyDataset, load_vocab, load_label_mapping, evaluate, Config
"""
    评估已经训练好的模型在测试集上的性能
    评估三个模型：random, glove_static, glove_finetune
"""


if __name__ == '__main__':
    # 配置和训练时一致 
    # embedding参数要和训练时用的一样
    config = Config('cnews_corpus', 'glove_static')
    # 设备和训练时一致
    config.device = torch.device('cpu')

    # 加载词汇表、构建测试数据集
    vocab_package = load_vocab(config.vocab_path)
    word2id = vocab_package['word2id']

    # 构建测试集DataLoader
    test_dataset = MyDataset(
        data_path=config.test_path,
        word2id=word2id,
        pad_size=config.pad_size,
        label2id=config.label2id
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=False  # 测试集不打乱
    )
    print(f"测试集样本数: {len(test_dataset)}")

    # 初始化模型 + 加载训练好的权重
    random_model = Model(config).to(config.device)
    glove_static_model = Model(config).to(config.device)
    glove_finetune_model = Model(config).to(config.device)
    # 最优权重路径
    random_path = r"Data/saved_dict/TextCNN_random.ckpt"
    glove_static_path = r"Data/saved_dict/TextCNN_glove_static.ckpt"
    glove_finetune_path = r"Data/saved_dict/TextCNN_glove_finetune.ckpt"
    # 加载最优权重
    random_state_dict = torch.load(random_path, map_location=config.device)
    random_model.load_state_dict(random_state_dict)
    glove_static_state_dict = torch.load(glove_static_path, map_location=config.device)
    glove_static_model.load_state_dict(glove_static_state_dict)
    glove_finetune_state_dict = torch.load(glove_finetune_path, map_location=config.device)
    glove_finetune_model.load_state_dict(glove_finetune_state_dict)
    print("模型权重加载完成")

    # 计算测试集准确率
    criterion = nn.CrossEntropyLoss()
    random_test_loss, random_test_acc = evaluate(random_model, test_loader, criterion, config.device)
    print("\n" + "="*50)
    print(f"随机模型的测试集损失：{random_test_loss:.4f}")
    print(f"随机模型的测试集准确率：{random_test_acc:.4f}  (即 {random_test_acc*100:.2f}%)")
    print("="*50)
    # 评估glove_static模型
    glove_static_test_loss, glove_static_test_acc = evaluate(glove_static_model, test_loader, criterion, config.device)
    print(f"glove_static模型的测试集损失：{glove_static_test_loss:.4f}")
    print(f"glove_static模型的测试集准确率：{glove_static_test_acc:.4f}  (即 {glove_static_test_acc*100:.2f}%)")
    print("="*50)
    # 评估glove_finetune模型
    glove_finetune_test_loss, glove_finetune_test_acc = evaluate(glove_finetune_model, test_loader, criterion, config.device)
    print(f"glove_finetune模型的测试集损失：{glove_finetune_test_loss:.4f}")
    print(f"glove_finetune模型的测试集准确率：{glove_finetune_test_acc:.4f}  (即 {glove_finetune_test_acc*100:.2f}%)")
    print("="*50)
