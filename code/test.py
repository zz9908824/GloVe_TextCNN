from os import path
import pickle
path = r"D:\研一下\python_project\TextCNN\Data\cnews_corpus\vocab.pkl"
package = pickle.load(open(path, 'rb'))
word2id = package['word2id']
id2word = package['id2word']
print(word2id['<pad>'])
print(word2id['<unk>'])
print(id2word[0])
print(id2word[1])
