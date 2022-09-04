from threading import Thread
import socket
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

class LocalServer(object):
    def __init__(self, host, port):
        self.address = (host, port)
        f = open('./data/vocab.txt')
        self.word_list = f.read().split('\n') # 词典
        f.close()
        self.orig_news = pd.read_csv('./data/all_news.csv') # 原新闻，最后返回的时候用
        self.tfidf = np.load('./data/tfidf_matrix.npy') # tfidf矩阵
        self.pca_tfidf = np.load('./data/pca.npy') # 降维后的tfidf矩阵
        self.pca_tfidf = normalize(self.pca_tfidf)
        self.news_keywords = np.load('./data/keywords.npy') # 每篇新闻的keywords
        f = open('./data/synonym.txt') # 每个词的近似词
        self.synonym = eval(f.read())
        return

    def search_data(self, datarecv):
        kwords = datarecv.lower().strip().split(' ')
        """
        主要思想：
        - 对检索的词语，找出他们的相关词。原词的相关度定为1，相关词的相关度定为两个词的cos-similarity
        - 这样得到词语的权重
        - 对文章，定义本次检索文章的相关度为上述词语向量乘以对应权重之和
        - 取前k个文章，并对文章做hits算法直到收敛
        - 结果中的前k篇文章作为检索结果
        """
        for word in kwords:
            if word not in self.word_list:
                kwords.remove(word)
        if len(kwords) == 0:
            return [('File Not Found!', 'No related news T_T')]

        # 对词语赋予权重。对查询的词，权重初始为1
        word_weight = dict()
        for word in kwords:
            id = self.word_list.index(word)
            word_weight[id] = 1.0
        # 考虑了近似词后的权重。相当于把近似词也都看作检索词，权重为cos-similarity
        for word in kwords:
            if word not in self.synonym:
                continue
            for syn in self.synonym[word]:
                id1 = self.word_list.index(word)
                id2 = self.word_list.index(syn)
                vec1 = normalize(self.tfidf[:, id1].reshape(1, -1)).reshape(-1)
                vec2 = normalize(self.tfidf[:, id2].reshape(1, -1)).reshape(-1)
                cos_sim = np.dot(vec1, vec2)
                if id2 not in word_weight:
                    word_weight[id2] = cos_sim
                else:
                    word_weight[id2] += cos_sim
        
        # 计算出所有文章关于这些词语的相关度，是一个文章数量维的向量
        news_weight = np.zeros_like(self.tfidf[:, 0])
        for id, weight in word_weight.items():
            vec = self.tfidf[:, id]*weight
            news_weight += vec
        
        news_weight = normalize(news_weight.reshape(1, -1)).reshape(-1)

        # 选取相关的文章
        selected = []
        score = []

        for id, weight in enumerate(news_weight):
            if weight > 0.1:
                selected.append(id)
                score.append(weight)
        score = np.array(score)
        # 下面考虑对于这些文章，若存在关键词与检索词重合，则说明两者主题相近，对其赋予一定的权重
        keywords_match = []

        for i in selected:
            matched = 0
            for word in word_weight:
                if word in self.news_keywords[i]:
                    matched += 1
            keywords_match.append(matched)
        keywords_match = np.array(keywords_match)

        # 下面考虑文章的相似文章，利用文章的cos-similarity
        sim_news = []
        for id in selected:
            sim_news.extend(self.find_sim_news(id))
        sim_news.extend(sim_news)
        sim_news = list(set(sim_news))
        sim_news = sorted(sim_news)
        sim_news = np.array(sim_news)

        # 对选取的文章、相似文章集合进行hits算法，得到文章的权重
        hits_weight = []

        sim_matrix = self.calc_sim_matrix(sim_news)
        eigv = self.calc_principal_eigenvector(sim_matrix, eps=1e-6)
        for i, id in enumerate(sim_news):
            # print('eigv:', eigv)
            # print('sim_mat[i]:', sim_matrix[i])
            weight = np.dot(eigv, sim_matrix[i])
            if id in selected:
                hits_weight.append(weight)
        
        # 最终的文章权重三部分组成：对于检索词汇的相关度，hits算法计算的权重，文章关键词的重合度。三者都是模长为1的向量，最后进行叠加
        fin_score = normalize(np.array(score).reshape(1, -1)).reshape(-1) + \
                normalize(np.array(hits_weight).reshape(1, -1)).reshape(-1) +\
                normalize(keywords_match.reshape(1, -1)).reshape(-1)

        # 根据计算的权重对文章排序并返回对应的数据
        selected = np.array(selected)
        selected = selected[np.argsort(-fin_score)]

        dataret = []
        for i, news in enumerate(selected):
            dataret.append((self.orig_news.iloc[news]['title'], self.orig_news.iloc[news]['body']))
        
        return dataret

    # 计算文章的cos-similarity矩阵
    def calc_sim_matrix(self, news):
        feature_matrix = self.pca_tfidf[news]
        return feature_matrix @ feature_matrix.T

    # 计算文章的相似文章
    def find_sim_news(self, id):
        cos_sim = np.dot(self.pca_tfidf, self.pca_tfidf[id])
        return list(np.argwhere(cos_sim > 0.1).reshape(-1)) # 这里的0.1正是之前数据处理时调参得到的参数
        
    # 计算矩阵的主特征向量，用于hits算法
    def calc_principal_eigenvector(self, mat, eps):
        v = np.random.rand(mat.shape[1])
        while True:
            tmp = np.dot(mat, v)
            tmp = tmp/np.linalg.norm(tmp)
            if np.abs(tmp-v).mean() < eps:
                break
            v = tmp
        return v

    # 处理用户端请求
    def handle_client(self, conn, addr):
        print(f'new thread created. Waiting requests from {addr}')

        datarecv = conn.recv(1000).decode('utf-8')
        print(f'receive {datarecv} from {addr}')
        result = self.search_data(datarecv)
        conn.send(str(result).encode('utf-8'))
        conn.close()

    def run(self):
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(self.address)
        server.listen(5)
        
        """
        TODO：请在服务器端实现合理的并发处理方案，使得服务器端能够处理多个客户端发来的请求
        """

        while True:
            conn, addr = server.accept()
            print(f'{conn} connected')
            t = Thread(target=self.handle_client, args=(conn, addr))
            t.start()
            
        """
        TODO: 请补充实现文本检索，以及服务器端与客户端之间的通信
        
        1. 接受客户端传递的数据， 例如检索词
        2. 调用检索函数，根据检索词完成检索
        3. 将检索结果发送给客户端，具体的数据格式可以自己定义
        
        """
        
server = LocalServer("127.0.0.1", 80)
server.run()