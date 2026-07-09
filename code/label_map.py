from signal import valid_signals
import matplotlib.pyplot as plt

def load_label_mapping(file_path):
    """
    读取标签映射文件，返回 {标签名: 编号} 的字典
    """
    label_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()  # 去除首尾空格、换行符
            if not line:
                continue  # 跳过空行
            # 按制表符 \t 分割，拆成标签名和编号
            label_name, label_id = line.split('\t')
            # 编号转成整数，存入字典
            label_dict[label_name] = int(label_id)
    return label_dict
def Label_map(corpus_path, label2id):
    """
    读取语料文件，把每行开头的标签转成数字id
    返回：processed_data = [(标签id, 文本内容), ...]
    """
    processed_data = []
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # 只拆分第一个空格，分离标签和文本，避免拆分后面的分词内容
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                print(f"警告：第{line_num}行格式异常，跳过：{line}")
                continue
            
            label_name, text = parts
            # 标签转数字id
            if label_name not in label2id:
                print(f"警告：第{line_num}行存在未知标签 {label_name}，跳过")
                continue
            
            label_id = label2id[label_name]
            processed_data.append((label_id, text))
    return processed_data
    
if __name__ == '__main__':
    train_path = r"D:\研一下\python_project\TextCNN\Data\cnews_corpus\corpus\cnews_train_corpus.txt"
    test_path = r"D:\研一下\python_project\TextCNN\Data\cnews_corpus\corpus\cnews_test_corpus.txt"
    val_path = r"D:\研一下\python_project\TextCNN\Data\cnews_corpus\corpus\cnews_val_corpus.txt"
    label_path = r"D:\研一下\python_project\TextCNN\Data\cnews_corpus\label_mapping.txt"
    label_dict = load_label_mapping(label_path)
    print(Label_map(train_path, label_dict))
    # Label_map(test_path, label_dict)
    # Label_map(val_path, label_dict)