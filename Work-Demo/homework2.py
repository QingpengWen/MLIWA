import jieba
from collections import Counter
# TODO: 测试分词前结果
text = '神里绫华大小姐是我老婆，可莉和纳西妲都是我和她的女儿。'
seg_list_before = jieba.cut(text)
print("/".join(seg_list_before))
# TODO: 加载自定义词典
jieba.load_userdict('./data2/my_dict.txt')
# TODO: 测试新词典的分词结果
seg_list_after = jieba.cut(text)
print("/".join(seg_list_after))
# TODO: 对句子进行分词
def seg_sentence(sentence):
    sentence_seged = jieba.cut(sentence.strip())
    outstr = ''
    for word in sentence_seged:
       if word != '\t':
            outstr += word
            outstr += "/"
    return outstr
inputs = open('./data2/story.txt', 'r', encoding='UTF-8') #加载要处理的文件的路径
outputs = open('./data2/output_before.txt', 'w')  # 加载处理后的文件路径
for line in inputs:
    line_seg = seg_sentence(line)  # 这里的返回值是字符串
    outputs.write(line_seg)
outputs.close()
inputs.close()
# WordCount
with open('./data2/output_before.txt', 'r') as fr:  # 读入分词后的文件
    data = jieba.cut(fr.read())
data = dict(Counter(data))
with open('./data2/cipin_before.txt', 'w') as fw:  # 读入存储wordcount的文件路径
    for k, v in data.items():
        fw.write('%s,%d\n' % (k, v))
