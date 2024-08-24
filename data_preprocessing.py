import networkx as nx
import dgl
from cdlib import algorithms
from torch_geometric.datasets import AttributedGraphDataset,WebKB,Twitch,LastFMAsia
import torch
from config import device
import pandas as pd
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, datapath, labelpath, dataset_name):
        self.data_path = datapath
        self.label_path=labelpath
        self.dataset_name = dataset_name
        self.load_data()
        self.preprocess_data()
        self.save_labels()
        self.create_splits()
        self.add_global_nodes()

    def load_data(self):
        if self.dataset_name=='cora' or self.dataset_name=='citeseer' or self.dataset_name=='wiki' or self.dataset_name=='pubmed':
            self.dataset = AttributedGraphDataset(self.data_path, name=self.dataset_name)
        if self.dataset_name=='LastFMAsia':
            self.dataset = LastFMAsia(self.data_path)
        if self.dataset_name=='PT':
            self.dataset = Twitch(self.data_path, name=self.dataset_name)
        self.data = self.dataset[0]
        self.number_class=self.dataset.num_classes

    def preprocess_data(self):
        self.G = nx.Graph()
        self.edge_index = self.data.edge_index.numpy()  # Store edge_index as an instance variable
        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[:, i]
            self.G.add_edge(src, dst, weight=1)
        self.g = dgl.from_networkx(self.G)
        self.node_features = self.data.x.numpy()
        self.default_feat = self.data.x
        
        self.coms = algorithms.wCommunity(self.G, min_bel_degree=0.26, threshold_bel_degree=0.26)

    def save_labels(self):
        with open(self.label_path, 'w') as f:
            for item in self.data.y.numpy():
                f.write("%s\n" % item)

    def create_splits(self, test_size=0.80):
        label_data = pd.read_csv(self.label_path, header=None)
        label = label_data[0].tolist()
        
        train_label, test_label = train_test_split(label, test_size=0.80, random_state=7)
        train_label=torch.tensor(train_label,dtype=torch.long).to(device)
        size=int(len(label)*0.1)
        val_label=train_label[0:size]
        train_label=train_label[size:]

        self.label=label
        self.train_label = torch.tensor(train_label, dtype=torch.long).to(device)
        self.val_label = torch.tensor(val_label, dtype=torch.long).to(device)
        self.test_label = torch.tensor(test_label, dtype=torch.long).to(device)

    
    def add_global_nodes(self):
        c = self.coms.communities
        self.coms_G = [nx.Graph(self.G.subgraph(c[i])) for i in range(len(c))]
        centrality = nx.closeness_centrality(self.G)
        '''
        ng stands for global node
        '''
        if self.dataset_name=='cora':
            ng=3
        if self.dataset_name=='citeseer':
            ng=1
        if self.dataset_name=='wiki':
            ng=4
        if self.dataset_name=='LastFMAsia':
            ng=4
        if self.dataset_name=='pubmed':
            ng=4    
        if self.dataset_name=='PT':
            ng=5
            
        newA = set(sorted(centrality, key=centrality.get, reverse=True)[:ng])
        for k in range(len(self.coms_G)):
            for i in list(self.coms_G[k].nodes()):
                for j in newA:
                    if i != j:
                        self.coms_G[k].add_edge(i, j)
        self.com_DICT = {i: list(self.coms_G[i].nodes) for i in range(len(self.coms_G))}
        
        self.egos_G = []
        self.ego_DICT = {}
        for node in sorted(self.G.nodes()):  # Ensure the nodes are processed in a sorted order
            ego_graph = nx.ego_graph(self.G, node, radius=1)
            self.egos_G.append(ego_graph)
            self.ego_DICT[node] = list(ego_graph.nodes())
            

    def get_data_components(self):
        return self.g,  self.G, self.com_DICT, self.coms_G, self.ego_DICT, self.egos_G, self.edge_index, self.default_feat,self.node_features,self.number_class
