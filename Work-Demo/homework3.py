import jieba
# TODO: 加载自定义词典
jieba.load_userdict('D:/学习资料/相关课程资料、答案/文本信息处理/TM/data2/my_dict.txt')

import requests
from bs4 import BeautifulSoup as be

md_url = "https://ys.mihoyo.com/content/ysCn/getContentList?pageSize=20&pageNum=1&order=asc&channelId=150"
ly_url = "https://ys.mihoyo.com/content/ysCn/getContentList?pageSize=20&pageNum=1&order=asc&channelId=151"
dq_url = "https://ys.mihoyo.com/content/ysCn/getContentList?pageSize=20&pageNum=1&order=asc&channelId=324"
xm_url = "https://ys.mihoyo.com/content/ysCn/getContentList?pageSize=20&pageNum=1&order=asc&channelId=350"
dir_path = './data2'

def get_json(_url_):
    req = requests.get(url=_url_)
    if req.status_code == 200:
        return req.json()['data']
    else:
        return None

def clean_data():
    returns = []
    for url in [md_url, ly_url, dq_url, xm_url]:
        _data_ = get_json(url)
        for key in _data_['list']:
            ext = key["ext"]
            data = [
                    key['title'],
                    be(ext[7]["value"], "lxml").p.text.strip(),
                    ]
            data_str = "".join(data)
            data_segments = jieba.cut(data_str)
            data_final = " ".join(data_segments)
            returns.append(data_final)
    return returns

def data():
    _json_ = clean_data()
    return _json_

# TODO:所有数据的集合
datasets = data()
documents = datasets
print(len(documents))

# 清理数据
import pandas as pd
news_df = pd.DataFrame({'document': documents})
# 去除字符\n
news_df['clean_doc'] = news_df['document'].str.replace("\n", " ")
print(news_df.head())
# 变为token
tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
# de-tokenization
detokenized_doc = []
for i in range(len(news_df)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)

news_df['clean_doc'] = detokenized_doc
print(news_df.head())

# TODO: 词汇表输出
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=67,
                             max_df=0.5,
                             smooth_idf=True)
# vectorizer = TfidfVectorizer(max_features=67, min_df=1, ngram_range=(1,1))
news_df_doc = news_df['clean_doc']
X = vectorizer.fit_transform(news_df_doc)
print(vectorizer.vocabulary_)
from sklearn.decomposition import TruncatedSVD
# SVD represent documents and terms in vectors
svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
svd_model.fit(X)
len(svd_model.components_)

# TODO: topics展示
terms = vectorizer.get_feature_names()
for i, comp in enumerate(svd_model.components_):
    terms_comp = zip(terms, comp)
    sorted_terms = sorted(terms_comp, key=lambda x: x[1], reverse=True)[:9]
    print("Topic "+str(i)+": ", end='')
    for t in sorted_terms:
        print(t[0], end='')
        print(" ", end='')
    print()

# TODO: UMAP可视化
import umap.umap_ as umap
import matplotlib.pyplot as plt
X_topics = svd_model.fit_transform(X)
embedding = umap.UMAP(n_neighbors=150, min_dist=0.5, random_state=12).fit_transform(X_topics)
plt.figure(figsize=(7, 5))
plt.title('UMAP Visualization')
plt.scatter(embedding[:, 0],
            embedding[:, 1],
            c=comp,
            s=80,
            edgecolor='none'
            )
plt.savefig('UMAP.png')
plt.show()

# TODO: PCA可视化
from sklearn.decomposition import PCA
embedding2 = PCA(n_components=2).fit_transform(X_topics)
plt.figure(figsize=(7, 5))
plt.title('PCA Visualization')
plt.scatter(embedding2[:, 0], embedding2[:, 1],
            c=comp,
            s=80,
            edgecolor='none'
            )
plt.savefig('PCA.png')
plt.show()

# TODO: t-SNE可视化
from sklearn.manifold import TSNE
embedding3 = TSNE(n_components=2).fit_transform(X_topics)
plt.figure(figsize=(7, 5))
plt.title('t-SNE Visualization')
plt.scatter(embedding3[:, 0], embedding3[:, 1],
            c=comp,
            s=80,
            edgecolor='none'
            )
plt.savefig('t-SNE.png')
plt.show()

