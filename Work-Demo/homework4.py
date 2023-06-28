import networkx as nx
import pandas as pd
import random
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# read dataset
df = pd.read_csv("./data4/yuansheng_data.tsv", sep="\t")
print(df.head())
print(df.shape)
# TODO:创建图节点
G = nx.from_pandas_edgelist(df, "城市", "角色名称", edge_attr=True, create_using=nx.Graph())
print(len(G))

# TODO:随机漫步算法
def get_randomwalk(node, path_length):
    random_walk = [node]
    for i in range(path_length - 1):
        temp = list(G.neighbors(node))
        temp = list(set(temp) - set(random_walk))
        if len(temp) == 0:
            break
        random_node = random.choice(temp)
        random_walk.append(random_node)
        node = random_node
    return random_walk
get_randomwalk('蒙德城', 10)
all_nodes = list(G.nodes())
random_walks = []
for n in tqdm(all_nodes):
    for i in range(5):
        random_walks.append(get_randomwalk(n, 10))
# count of sequences
print(len(random_walks))

# TODO:模型训练
import gensim.models as GSmodel
import warnings
warnings.filterwarnings('ignore')
# train word2vec model
model = GSmodel.Word2Vec(window=4, sg=1, hs=0,
                 negative=10,  # for negative sampling
                 alpha=0.03, min_alpha=0.0007,
                 seed=14)
model.build_vocab(random_walks, progress_per=2)
model.train(random_walks, total_examples=model.corpus_count, epochs=20, report_delay=1)
print(model)
# find top n similar nodes
word = '琴'
GSmodel.Word2Vec.similar_by_word(model, word)

# TODO:PCA主成分分析获得原神各角色名字的相似度并绘图可视化展示
def plot_nodes(word_list):
    X = model[word_list]
    # reduce dimensions to 2
    pca = PCA(n_components=2)
    result = pca.fit_transform(X)
    plt.figure(figsize=(12, 9))
    # create a scatter plot of the projection
    plt.scatter(result[:, 0], result[:, 1])
    for i, word in enumerate(word_list):
        plt.annotate(word, xy=(result[i, 0], result[i, 1]))
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    plt.rcParams['axes.unicode_minus'] = False
    plt.title('原神各角色名字相似度')
    plt.savefig('原神各角色名字相似度.png')
    plt.show()

terms = ['琴', '安柏', '丽莎', '凯亚',
         '芭芭拉', '迪卢克', '雷泽', '温迪',
         '可莉', '班尼特', '诺艾尔', '菲谢尔',
         '砂糖', '莫娜', '迪奥娜', '阿贝多',
         '罗莎莉亚', '优菈', '埃洛伊', '米卡',
         '魈', '北斗', '凝光', '香菱',
         '行秋', '重云', '刻晴', '七七',
         '达达利亚', '钟离', '辛焱', '甘雨',
         '胡桃', '烟绯', '申鹤', '云堇',
         '夜兰', '瑶瑶', '白术', '神里绫华',
         '枫原万叶', '宵宫', '早柚', '雷电将军',
         '九条裟罗', '珊瑚宫心海', '托马', '荒泷一斗',
         '五郎', '八重神子', '神里绫人', '久岐忍',
         '鹿野院平藏', '绮良良', '提纳里', '柯莱',
         '多莉', '赛诺', '坎蒂丝', '妮露',
         '纳西妲', '莱依拉', '流浪者', '珐露珊',
         '艾尔海森', '迪希雅', '卡维']
plot_nodes(terms)
