# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import os
import random
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from label_map import load_label_mapping


def set_seed(seed=42):
    """固定随机种子 可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_vocab(path):
    """加载词汇表【package（word2id, id2word）】，增加异常处理"""
    try:
        with open(path, 'rb') as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"词汇表文件不存在: {path}")
    except Exception as e:
        raise RuntimeError(f"词汇表加载失败: {str(e)}")
    
    if 'word2id' not in vocab or 'id2word' not in vocab:
        raise ValueError("词汇表格式错误，缺少 word2id 或 id2word 字段")
    return vocab


def load_glove_txt(path):
    """加载文本格式的词向量文件【glove预训练好的词向量文件】，增加容错处理"""
    glove = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue  # 跳过格式错误的行
                word = parts[0]
                vec = np.array([float(x) for x in parts[1:]], dtype='float32')
                glove[word] = vec
    except FileNotFoundError:
        raise FileNotFoundError(f"词向量文件不存在: {path}")
    
    if not glove:
        raise ValueError(f"词向量文件为空或格式全部错误: {path}")
    return glove


class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextCNN'
        # 数据集路径（适配已转数字标签的分词文件）
        self.train_path = r'D:\研一下\python_project\TextCNN\Data\cnews_corpus\corpus\cnews_train_corpus.txt'
        self.dev_path = r'D:\研一下\python_project\TextCNN\Data\cnews_corpus\corpus\cnews_val_corpus.txt'
        self.test_path = r'D:\研一下\python_project\TextCNN\Data\cnews_corpus\corpus\cnews_test_corpus.txt'
        self.label_path = r'D:\研一下\python_project\TextCNN\Data\cnews_corpus\label_mapping.txt'
        self.vocab_path = r'D:\研一下\python_project\TextCNN\Data\cnews_corpus\vocab.pkl'
        self.save_path = rf'D:\研一下\python_project\TextCNN\Data\saved_dict\TextCNN_{embedding}.ckpt'
        self.log_path = rf'D:\研一下\python_project\TextCNN\Data\log\TextCNN_{embedding}.log'
        # gpu torch不兼容不能使用gpu，只能使用cpu
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dropout = 0.5
        self.early_stop_patience = 5  # 早停容忍轮次（epoch级）
        # 从标签映射文件自动获取类别数，避免硬编码不一致
        self.label2id = load_label_mapping(self.label_path)
        self.num_classes = len(self.label2id)
        self.embedding_mode = embedding
        
        self.num_epochs = 20
        self.batch_size = 128
        self.pad_size = 500
        self.learning_rate = 5e-4
        self.filter_sizes = (2, 3, 4)
        self.num_filters = 256
        
        # 预训练词向量加载逻辑（三种模式：基线random / glove_static / glove_finetune）
        if embedding.startswith('glove'):
            # 所有GloVe模式都加载预训练词向量
            vec_path = r'd:\研一下\python_project\TextCNN\Data\vector\vectors.txt'
            self.glove = load_glove_txt(vec_path)
            # 自动获取词向量维度
            sample_vec = next(iter(self.glove.values()))
            self.embed = len(sample_vec)
            # static模式冻结嵌入层，finetune模式不冻结
            self.freeze_embedding = (embedding == 'glove_static')
        else:
            # 随机初始化模式
            self.glove = None
            self.freeze_embedding = False
            self.embed = 100  


# 自定义数据集类（中文标签+分词文本格式）
class MyDataset(Dataset):
    def __init__(self, data_path, word2id, pad_size=500,label_map=None):
        self.texts = []
        self.labels = []
        
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 按第一个空格拆分：数字标签 + 文本
                parts = line.split(maxsplit=1)
                if len(parts) < 2:
                    continue
                label_id, text = parts
                # 标签直接转整数
                self.labels.append(label_map[label_id])
                
                # 文本转词索引
                words = text.split()
                words_id = [word2id.get(w, word2id['<unk>']) for w in words]
                # 截断/补全到固定长度
                if len(words_id) >= pad_size:
                    words_id = words_id[:pad_size]
                else:
                    words_id += [word2id['<pad>']] * (pad_size - len(words_id))
                self.texts.append(torch.LongTensor(words_id))
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class Model(nn.Module):
    """Convolutional Neural Networks for Sentence Classification"""
    def __init__(self, config):
        super(Model, self).__init__()
        # 加载词表，获取真实词表大小
        vocab_package = load_vocab(config.vocab_path)
        word2id = vocab_package['word2id']
        vocab_size = len(word2id)
        
        # 分分支构建嵌入层
        if config.glove is not None:
            # 预训练模式：按索引对齐填充词向量
            embedding_matrix = np.zeros((vocab_size, config.embed), dtype='float32')
            glove_vec = config.glove
            for word, idx in word2id.items():
                if word == '<pad>':
                    continue  # pad 保持全0，不参与语义计算
                elif word in glove_vec:
                    embedding_matrix[idx] = glove_vec[word]
                else:
                    # OOV 词小范围随机初始化，利于收敛
                    embedding_matrix[idx] = np.random.uniform(-0.05, 0.05, config.embed)
            # 加载预训练权重，根据配置决定是否冻结（修复冻结失效问题）
            self.embedding = nn.Embedding.from_pretrained(
                torch.tensor(embedding_matrix, dtype=torch.float32),
                freeze=config.freeze_embedding,
                padding_idx=0
            )
        else:
            # 随机初始化模式
            self.embedding = nn.Embedding(
                num_embeddings=vocab_size,
                embedding_dim=config.embed,
                padding_idx=0
            )
        
        # 多尺度卷积层
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes]
        )
        # 正则化与分类层
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)
    
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x
    
    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


def evaluate(model, data_loader, criterion, device):
    """评估函数：计算数据集上的损失和准确率"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            output = model(batch_x)
            loss = criterion(output, batch_y)
            
            total_loss += loss.item()
            pred = torch.argmax(output, dim=1)
            correct += torch.sum(pred == batch_y).item()
            total += len(batch_y)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == '__main__':
    # 固定随机种子
    set_seed(42)
    
    # 环境信息打印
    print(f"CUDA可用: {torch.cuda.is_available()}")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"PyTorch版本: {torch.__version__}")
    # 配置参数 model模式：random/glove_finetune/glove_static
    config = Config('cnews_corpus', 'glove_finetune')
    print(f"\n运行设备: {config.device}")
    print(f"嵌入模式: {config.embedding_mode}")
    print(f"词向量维度: {config.embed}")
    print(f"类别数: {config.num_classes}")
    
    model = Model(config).to(config.device)
    print("\n模型结构:")
    print(model)
    
    # 测试前向传播
    test_input = torch.randint(0, 1000, (2, config.pad_size)).to(config.device)
    output = model(test_input)
    print(f"\n前向传播输出形状: {output.shape}")
    print("模型初始化与前向传播全部正常\n")
    
    # 加载词汇表（仅加载一次，复用给数据集）
    vocab_package = load_vocab(config.vocab_path)
    word2id = vocab_package['word2id']
    
    # 创建数据集与DataLoader
    train_dataset = MyDataset(config.train_path, word2id, config.pad_size,config.label2id)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False
    )
    dev_dataset = MyDataset(config.dev_path, word2id, config.pad_size,config.label2id)
    dev_loader = DataLoader(dev_dataset, batch_size=config.batch_size, shuffle=False)
    test_dataset = MyDataset(config.test_path, word2id, config.pad_size,config.label2id)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(dev_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}\n")
    
    # 自动创建保存目录
    os.makedirs(os.path.dirname(config.save_path), exist_ok=True)
    
    # 损失函数、优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_acc = 0
    last_improve_epoch = 0
    train_loss_list = []
    dev_loss_list = []
    dev_acc_list = []
    
    for epoch in range(config.num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(config.device)
            batch_y = batch_y.to(config.device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        # 计算平均损失
        avg_train_loss = total_train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        
        # 验证集评估
        dev_loss, dev_acc = evaluate(model, dev_loader, criterion, config.device)
        dev_loss_list.append(dev_loss)
        dev_acc_list.append(dev_acc)
        
        # 打印日志
        print(f"Epoch [{epoch+1}/{config.num_epochs}]")
        print(f"  训练损失: {avg_train_loss:.4f} | 验证损失: {dev_loss:.4f} | 验证准确率: {dev_acc:.4f}")
        
        # 保存最优模型 + 早停判断
        if dev_acc > best_acc:
            best_acc = dev_acc
            last_improve_epoch = epoch
            torch.save(model.state_dict(), config.save_path)
            print(f"  最佳模型已保存，验证准确率: {best_acc:.4f}")
        # 早停：超过容忍轮次没提升则终止训练（使用配置参数）
        elif epoch - last_improve_epoch >= config.early_stop_patience:
            print(f"  连续{config.early_stop_patience}轮验证准确率未提升，提前终止训练")
            break
    
    print(f"\n训练完成！最佳验证准确率: {best_acc:.4f}")
    # 加载最优模型，在测试集上最终评估
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, config.device)
    print(f"测试集最终结果：损失 {test_loss:.4f} | 准确率 {test_acc:.4f}")
    
    # 绘制损失+准确率曲线（适配早停后的实际长度）
    epochs = range(1, len(train_loss_list) + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss_list, 'b-', label='Train Loss')
    plt.plot(epochs, dev_loss_list, 'r-', label='Dev Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{config.embedding_mode} Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, dev_acc_list, 'g-', label='Dev Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    import os
    plot_dir = r'D:\研一下\python_project\TextCNN\Data\plots'
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f'TextCNN_{config.embedding_mode}_plot.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n训练曲线已保存到: {plot_path}")
    
    try:
        plt.show()
    except Exception as e:
        print(f"图形显示失败（可能无图形界面）: {e}")
        print("请直接查看保存的图片文件")
