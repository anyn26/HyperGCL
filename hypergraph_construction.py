import torch
import torch.nn as nn
import numpy as np
import dgl
from sklearn.cluster import KMeans
import faiss
import networkx as nx
import torch.nn.functional as F
from data_preprocessing import DataProcessor
from config import device
import copy
from collections import defaultdict
from numpy import linalg as LA

'''
HyG1 using k-nn, k-means both
'''
kmeans_k=50
knn_k = 60
S =  2
LEN1=0

class LearnableMaskMatrix_for_knn_kmeans_based_hyG1(nn.Module):
    def __init__(self, num_hyperedges, num_nodes, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.logits = nn.Parameter(torch.randn(num_nodes, num_hyperedges+kmeans_k))
        self.gradient_map = torch.zeros_like(self.logits)

    def capture_gradients(self):
        '''
        Function to capture gradients for the logits
        '''
        def hook(grad):
            self.gradient_map = grad.clone()
        self.logits.register_hook(hook)

    def forward(self, knn_kmeans,num_nodes):
        
        '''
        if U is a uniform random variable on (0, 1), then -log(-log(U)) transforms it into a Gumbel distribution.
        This transformation is used to add Gumbel noise to the logits.
        
        In the below, torch.rand(self.logits.size(), device=self.logits.device))--> is uniformly distributed between (0,1)
        '''
        #gumbel_noise=0
        gumbel_noise = -torch.log(-torch.log(torch.rand(self.logits.size(), device=self.logits.device)))
        y_soft = F.softmax((self.logits + gumbel_noise) / self.temperature, dim=-1)
        
        '''
        A hard threshold is applied to y_soft to create y_hard, 
        which is a binary matrix indicating whether a node belongs to a hyperedge/cluster.
        '''
        y_hard = (y_soft > 0.8).float()
        
        '''
        By subtracting y_soft from y_hard and then adding back y_soft, 
        we are performing a kind of "correction" to the gradients. 
        
        (y_hard - y_soft).detach(), we use detach() so that y_hard part doesn't participate in the backpropagation part.
        However, we add y_soft, that takes part in backpropagation. (this is a trick)
        '''
        mask_prob =  (y_hard - y_soft).detach()+y_soft 
        knn_kmeans_matrix = np.zeros((num_nodes, len(knn_kmeans)))
        
        '''
        Store original hyperedges for each node
        '''
        original_hyperedges_of_nodes = {}

        for col, values in knn_kmeans.items():
            length = len(values)
            knn_kmeans_matrix[:length, col] = values
            for node in values:
                if node not in original_hyperedges_of_nodes:
                    original_hyperedges_of_nodes[node] = set()
                original_hyperedges_of_nodes[node].add(col)

        '''
        Convert knn_kmeans_matrix to a PyTorch tensor
        '''
        knn_kmeans_tensor = torch.tensor(knn_kmeans_matrix, dtype=mask_prob.dtype, device=mask_prob.device)
       
        
        '''
        Element-wise multiplication
        '''
        final_matrix =  knn_kmeans_tensor*y_hard

        '''
        Convert the final matrix back to dictionary and find missing nodes
        '''
        knn_kmeans_mask_dict = {}
        nodes_in_hyperedges = set()

        for col in range(final_matrix.shape[1]):
            nodes = final_matrix[:, col].nonzero(as_tuple=False).view(-1).tolist()
            knn_kmeans_mask_dict[col] = nodes
            nodes_in_hyperedges.update(nodes)

        all_nodes = set(range(num_nodes))
        missing_nodes = all_nodes - nodes_in_hyperedges

        '''
        Restore missing nodes to their original hyperedges
        '''
        for node in missing_nodes:
            original_hyperedges = original_hyperedges_of_nodes.get(node, [])
            for hyperedge in original_hyperedges:
                if hyperedge in knn_kmeans_mask_dict:
                    knn_kmeans_mask_dict[hyperedge].append(node)
                    #break
        return knn_kmeans_mask_dict



def hyG1_function(node_features,knn_kmeans_mask_module,G):

    if isinstance(node_features, torch.Tensor):
          if node_features.is_cuda:
              node_features = node_features.detach().cpu().numpy()
          else:
              node_features = node_features.detach().numpy()

    kmeans = KMeans(n_clusters=kmeans_k).fit(node_features)
    C = kmeans.cluster_centers_
    knn_kmeans = {}
    
    '''
    FAISS CPU setup
    '''
    d = node_features.shape[1]
    index_flat = faiss.IndexFlatL2(d)
    index_flat.add(node_features)

    for idx, v in enumerate(node_features):
        # 3. Obtain k-nearest neighbors to node v (including itself)
        _, neighbors_indices = index_flat.search(np.array([v]), knn_k)
        knn_kmeans[idx] = list(neighbors_indices[0])

    new_keys = range(max(knn_kmeans.keys()) + 1, max(knn_kmeans.keys()) + 1 + kmeans_k)
    for key in new_keys:
        knn_kmeans[key] = []
    '''
    FAISS CPU setup for cluster centers
    '''
    index_C_flat = faiss.IndexFlatL2(d)
    index_C_flat.add(C)

    for idx, v in enumerate(node_features):
        # 5. Calculate Euclidean distance between k cluster centers and node v
        _, sorted_cluster_indices = index_C_flat.search(np.array([v]), S-1)

        # Assign v to its adjacent_clusters in the knn_kmeans dictionary
        for cluster_idx in sorted_cluster_indices[0]:
            key = max(knn_kmeans.keys()) - kmeans_k + 1 + cluster_idx
            knn_kmeans[key].append(idx)
    
    num_nodes = G.number_of_nodes()
    knn_kmeans = knn_kmeans_mask_module(knn_kmeans,num_nodes)
    DICT=knn_kmeans
    LEN1=len(DICT)
    n_hedge = LEN1

    member_citing = []
    for community in DICT.keys():
        members = DICT[community]
        for member in members:
            member_citing.append([member, community])

    member_community = torch.LongTensor(member_citing)
    data_dict = {
    ('node', 'in', 'edge'): (member_community[:, 0], member_community[:, 1]),
    ('edge', 'con', 'node'): (member_community[:, 1], member_community[:, 0])
    }

    '''
    Calculate the number of nodes considering both unique indices and the highest index
    '''
    unique_nodes = set(member_community[:, 0].tolist())
    max_node_index = max(unique_nodes)
    num_nodes = max(len(unique_nodes), max_node_index + 1)  # Ensure it covers all indices

    num_nodes_dict = {'edge': LEN1, 'node': num_nodes}
    hyG = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    rows=num_nodes
    columns=n_hedge


    len_rows=rows
    nl=np.eye(len_rows)
    nl=torch.from_numpy(nl)
    v_feat=nl

    len_edges=n_hedge
    nl=np.eye(len_edges)
    nl=torch.from_numpy(nl)
    e_feat=nl

    hyG.ndata['h'] = {'edge' : e_feat.type('torch.FloatTensor'), 'node' : v_feat.type('torch.FloatTensor')}
    e_feat = e_feat.type('torch.FloatTensor')
    v_feat=v_feat.type('torch.FloatTensor')

    v_feat1=v_feat.to(device)
    e_feat1=e_feat.to(device)
    hyG1=hyG.to(device)

    return v_feat1 ,e_feat1, hyG1, LEN1, DICT



class LearnableMaskMatrix_for_natural_hyG2(nn.Module):
    def __init__(self, num_hyperedges, num_nodes, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.logits = nn.Parameter(torch.randn(num_nodes, num_hyperedges))
        self.gradient_map = torch.zeros_like(self.logits)
    
    def capture_gradients(self):
        '''
        Function to capture gradients for the logits
        '''
        def hook(grad):
            self.gradient_map = grad.clone()
        self.logits.register_hook(hook)
    '''
    def update_gradient_map(self, new_gradients):
        # Update the gradient map, for example by averaging with new gradients
        self.gradient_map = (self.gradient_map + new_gradients) / 2
    '''   
    
    def forward(self, DICT,num_nodes,egos_G):
        '''
        Iterate over each key (hyperedge) and its values (nodes) and convert the DICT to a com_dict_matrix
        '''
        ego_dict_matrix = np.zeros((num_nodes, num_nodes))
        for key, values in DICT.items(): 
            for value in values:
                ego_dict_matrix[value, key] = 1  # Assuming 0-based indexing
        '''
        Populate original_hyperedges_of_nodes
        '''
        original_hyperedges_of_nodes = {}
        for col, values in DICT.items():
            for node in values:
                if node not in original_hyperedges_of_nodes:
                    original_hyperedges_of_nodes[node] = set()
                original_hyperedges_of_nodes[node].add(col)
        
        #gumbel_noise=0
        gumbel_noise = -torch.log(-torch.log(torch.rand(self.logits.size(), device=self.logits.device)))
        y_soft = F.softmax((self.logits + gumbel_noise) / self.temperature, dim=-1)
        y_hard = (y_soft > 0.8).float() # Hard sampling for forward pass only
        mask_prob = (y_hard - y_soft).detach()+y_soft

        #max_length = max(len(v) for v in DICT.values())
        ego_dict_tensor = torch.tensor(ego_dict_matrix, dtype=mask_prob.dtype, device=mask_prob.device) # Convert com_dict_matrix to a PyTorch tensor
        #mask_prob = mask_prob[:max_length, :]
        ego_dict_mask_tensor = ego_dict_tensor * y_hard  # Element-wise multiplication

        '''
        Converting com_dict_mask_tensor to a dictionary again
        '''
        ego_dict_mask_dict = {col: ego_dict_mask_tensor[:, col].nonzero(as_tuple=False).view(-1).tolist()
                              for col in range(ego_dict_mask_tensor.shape[1])}

        '''
        Restore missing nodes to their original hyperedges
        '''
        actual_all_nodes = set(range(num_nodes))
        nodes_in_new_hyperedges = set.union(*map(set, ego_dict_mask_dict.values()))
        missing_nodes_in_new_hypergraph = actual_all_nodes - nodes_in_new_hyperedges

        for node in missing_nodes_in_new_hypergraph:
            original_hyperedges = original_hyperedges_of_nodes.get(node, [])
            for hyperedge in original_hyperedges:
                if hyperedge in ego_dict_mask_dict:
                    ego_dict_mask_dict[hyperedge].append(node)
                    #break

        egos_G_refined=[]
        demo_egos_G=copy.deepcopy(egos_G)
        for k in range(len(demo_egos_G)):
            ego_graph_demo=nx.Graph()
            ego_graph_demo=copy.deepcopy(demo_egos_G[k])
            
            updated_nodes = ego_dict_mask_dict[k] # Get the updated nodes list for this community from com_dict_mask_dict

            '''
            Get the current nodes in the community graph
            '''
            current_nodes = ego_graph_demo.nodes()
            
            '''
            Determine nodes to be removed (present in current_nodes but not in updated_nodes)
            '''
            nodes_to_remove = current_nodes - updated_nodes

            '''
            Remove these nodes from the community graph
            '''
            for node in nodes_to_remove:
                ego_graph_demo.remove_node(node)
            egos_G_refined.append(ego_graph_demo)
        ego_dict_refined=ego_dict_mask_dict
        return ego_dict_refined,egos_G_refined


# Then, in the hyG2_function:
def hyG2_function(g, ego_DICT, egos_G, num_nodes, natural_hyG_mask_module):
    DICT = ego_DICT
    DICT, egos_G_refined = natural_hyG_mask_module(DICT, num_nodes, egos_G)
    LEN2 = len(DICT)
    n_hedge = LEN2

    member_citing = []
    for community in DICT.keys():
        members = DICT[community]
        for member in members:
            member_citing.append([member, community])

    member_community = torch.LongTensor(member_citing)
    data_dict = {
        ('node', 'in', 'edge'): (member_community[:, 0], member_community[:, 1]),
        ('edge', 'con', 'node'): (member_community[:, 1], member_community[:, 0])
    }

    unique_nodes = set(member_community[:, 0].tolist())
    max_node_index = max(unique_nodes)
    num_nodes_dict = {'edge': LEN2, 'node': max(len(unique_nodes), max_node_index + 1)}

    hyG = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)
    
    len_rows = g.number_of_nodes()
    v_feat = torch.eye(len_rows)

    len_edges = n_hedge
    e_feat = torch.eye(len_edges)

    hyG.ndata['h'] = {'edge': e_feat.type(torch.FloatTensor), 'node': v_feat.type(torch.FloatTensor)}
    e_feat = e_feat.type(torch.FloatTensor)
    v_feat = v_feat.type(torch.FloatTensor)

    '''
    Find the number of keys each value appears in
    '''
    value_counts = defaultdict(int)
    for key, values in DICT.items():
        unique_values = set(values)
        for value in unique_values:
            value_counts[value] += 1
    '''
    Calculate the score for each value
    '''
    total_keys = len(DICT)
    value_scores = {}
    uniqueness_list=[]
    for value, count in value_counts.items():
        score = 1 - (count / total_keys)
        value_scores[value] = score
        uniqueness_list.append(score)

    uniqueness = np.array(uniqueness_list)
    uniqueness = torch.LongTensor(uniqueness).to(device)

    v_feat2 = v_feat.to(device)
    e_feat2 = e_feat.to(device)
    hyG2 = hyG.to(device)

    return v_feat2, e_feat2, hyG2, LEN2, egos_G_refined, DICT, uniqueness



class LearnableMaskMatrix_for_structure_based_hyG3(nn.Module):
    def __init__(self, num_hyperedges, num_nodes, temperature=0.2):
        super().__init__()
        self.temperature = temperature
        self.logits = nn.Parameter(torch.randn(num_nodes, num_hyperedges))
        self.gradient_map = torch.zeros_like(self.logits)
    
    def capture_gradients(self):
        '''
        Function to capture gradients for the logits
        '''
        def hook(grad):
            self.gradient_map = grad.clone()
        self.logits.register_hook(hook)
    '''
    def update_gradient_map(self, new_gradients):
        # Update the gradient map, for example by averaging with new gradients
        self.gradient_map = (self.gradient_map + new_gradients) / 2
    '''   
    
    def forward(self, DICT,num_nodes,num_hyperedges,coms_G):
        '''
        Iterate over each key (hyperedge) and its values (nodes) and convert the DICT to a com_dict_matrix
        '''
        com_dict_matrix = np.zeros((num_nodes, num_hyperedges))
        for key, values in DICT.items(): 
            for value in values:
                com_dict_matrix[value, key] = 1  # Assuming 0-based indexing
        '''
        Populate original_hyperedges_of_nodes
        '''
        original_hyperedges_of_nodes = {}
        for col, values in DICT.items():
            for node in values:
                if node not in original_hyperedges_of_nodes:
                    original_hyperedges_of_nodes[node] = set()
                original_hyperedges_of_nodes[node].add(col)
        
        #gumbel_noise=0
        gumbel_noise = -torch.log(-torch.log(torch.rand(self.logits.size(), device=self.logits.device)))
        y_soft = F.softmax((self.logits + gumbel_noise) / self.temperature, dim=-1)
        y_hard = (y_soft > 0.8).float() # Hard sampling for forward pass only
        mask_prob = (y_hard - y_soft).detach()+y_soft

        #max_length = max(len(v) for v in DICT.values())
        com_dict_tensor = torch.tensor(com_dict_matrix, dtype=mask_prob.dtype, device=mask_prob.device) # Convert com_dict_matrix to a PyTorch tensor
        #mask_prob = mask_prob[:max_length, :]
        com_dict_mask_tensor = com_dict_tensor * y_hard  # Element-wise multiplication

        '''
        Converting com_dict_mask_tensor to a dictionary again
        '''
        com_dict_mask_dict = {col: com_dict_mask_tensor[:, col].nonzero(as_tuple=False).view(-1).tolist()
                              for col in range(com_dict_mask_tensor.shape[1])}

        '''
        Restore missing nodes to their original hyperedges
        '''
        actual_all_nodes = set(range(num_nodes))
        nodes_in_new_hyperedges = set.union(*map(set, com_dict_mask_dict.values()))
        missing_nodes_in_new_hypergraph = actual_all_nodes - nodes_in_new_hyperedges

        for node in missing_nodes_in_new_hypergraph:
            original_hyperedges = original_hyperedges_of_nodes.get(node, [])
            for hyperedge in original_hyperedges:
                if hyperedge in com_dict_mask_dict:
                    com_dict_mask_dict[hyperedge].append(node)
                    #break

        coms_G_refined=[]
        demo_coms_G=copy.deepcopy(coms_G)
        for k in range(len(demo_coms_G)):
            com_graph_demo=nx.Graph()
            com_graph_demo=copy.deepcopy(demo_coms_G[k])
            
            updated_nodes = com_dict_mask_dict[k] # Get the updated nodes list for this community from com_dict_mask_dict
            '''
            Get the current nodes in the community graph
            '''
            current_nodes = com_graph_demo.nodes()
            '''
            Determine nodes to be removed (present in current_nodes but not in updated_nodes)
            '''
            nodes_to_remove = current_nodes - updated_nodes
            '''
            Remove these nodes from the community graph
            '''
            for node in nodes_to_remove:
                com_graph_demo.remove_node(node)
            coms_G_refined.append(com_graph_demo)
        com_dict_refined=com_dict_mask_dict
        return com_dict_refined,coms_G_refined



#HyG3
def hyG3_function(g,com_DICT,coms_G,num_nodes,structural_hyG_mask_module):
    DICT=copy.deepcopy(com_DICT)
    LEN3=len(DICT)
    n_hedge = LEN3

    DICT,coms_G_refined=structural_hyG_mask_module(DICT,num_nodes,n_hedge,coms_G)
    member_citing = []
    
    for community in DICT.keys():
        members = DICT[community]
        for member in members :
            member_citing.append([member, community])

    member_community = torch.LongTensor(member_citing)
    data_dict = {
            ('node', 'in', 'edge'): (member_community[:,0], member_community[:,1]),
            ('edge', 'con', 'node'): (member_community[:,1], member_community[:,0])
        }

    lst=[]
    for i in member_citing:
      lst.append(i[0])
    s=set(lst)
    s=len(s)
    num_nodes_dict = {'edge': LEN3,'node':s}
    hyG = dgl.heterograph(data_dict,num_nodes_dict=num_nodes_dict)
    rows=g.number_of_nodes()
    columns=n_hedge


    len_rows=rows
    nl=np.eye(len_rows)
    nl=torch.from_numpy(nl)
    v_feat=nl

    len_edges=n_hedge
    nl=np.eye(len_edges)
    nl=torch.from_numpy(nl)
    e_feat=nl


    hyG.ndata['h'] = {'edge' : e_feat.type('torch.FloatTensor'), 'node' : v_feat.type('torch.FloatTensor')}
    e_feat = e_feat.type('torch.FloatTensor')
    v_feat=v_feat.type('torch.FloatTensor')

    '''
    Find the number of keys each value appears in
    '''
    value_counts = defaultdict(int)
    for key, values in DICT.items():
        unique_values = set(values)
        for value in unique_values:
            value_counts[value] += 1
    '''
    Calculate the score for each value
    '''
    total_keys = len(DICT)
    value_scores = {}
    uniqueness_list=[]
    for value, count in value_counts.items():
        score = 1 - (count / total_keys)
        value_scores[value] = score
        uniqueness_list.append(score)

    uniqueness = np.array(uniqueness_list)
    uniqueness = torch.LongTensor(uniqueness).to(device)

    v_feat3=v_feat.to(device)
    e_feat3=e_feat.to(device)
    hyG3=hyG.to(device)
    
    return v_feat3 ,e_feat3, hyG3, LEN3,coms_G_refined,DICT,uniqueness