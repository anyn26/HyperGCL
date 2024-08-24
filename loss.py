import torch
import torch.nn.functional as F
import numpy as np
from dgl import DGLGraph
from config import device


'''
CL Loss regular one
'''
def cosine_similarity(x, y):
    return torch.mm(x, y.t()) / (x.norm(dim=1).unsqueeze(1) * y.norm(dim=1).unsqueeze(0))


# Create a mask for first-order neighbors
def create_neighborhood_mask(num_nodes, edge_index):
    adjacency_matrix = torch.eye(num_nodes)# Initialize the adjacency matrix with self-loops
    '''
    Fill in the adjacency matrix with edges
    edge_index should be a tensor of shape [2, num_edges] where the first row contains source nodes
    and the second row contains target nodes.
    '''
    adjacency_matrix[edge_index[0], edge_index[1]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t() # Assuming undirected graph, make the adjacency matrix symmetric
    adjacency_matrix[adjacency_matrix > 1] = 1
    return adjacency_matrix.bool()



'''
Positive: a node and it's neighbors OR two nodes belong to same HyE. 
Negative: else
'''
def combined_contrastive_loss(Hx, Hy, hypergraph_dict, edge_index,  tau=0.5,k=50):
    num_nodes = Hx.size(0)
    sim_matrix = cosine_similarity(Hx, Hy)

    neighborhood_mask = create_neighborhood_mask(num_nodes, edge_index).to(Hx.device)
    hyperedge_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=Hx.device)
    for hyperedge in hypergraph_dict.values():
        for i in hyperedge:
            for j in hyperedge:
                #if i != j:
                    hyperedge_mask[i, j] = 1

    combined_pos_mask = neighborhood_mask | hyperedge_mask # Combine masks for positive samples
   
    contrastive_loss_values = torch.zeros(num_nodes, device=Hx.device)
    epsilon = 1e-8  # Small constant to prevent division by zero or log of zero

    # Compute the contrastive loss for each node
    for i in range(num_nodes):
        pos_mask = combined_pos_mask[i]
        pos_sim = sim_matrix[i][pos_mask]
        
        #identity_mask = torch.eye(num_nodes, device=Hx.device).bool() 
        #neg_mask = ~(pos_mask | identity_mask[i]) # Negative similarities: all other nodes excluding self and positive samples
        #neg_mask = ~(pos_mask)


        '''
        path/distance-based
        '''
        
        distance_from_i = shortest_paths_tensor[i]
        distances, sorted_indices = torch.sort(distance_from_i, descending=True)# Descending order sorting to get the most distance nodes first.
        neg_indices_mask = ~pos_mask[sorted_indices]
        selected_neg_indices = sorted_indices[neg_indices_mask][:k]
        
        
        '''
        similarity-based 
        '''
        '''
        sim_scores_with_self_excluded = cosine_similarity_matrix[i].clone()
        sim_scores_with_self_excluded[i] = -1.0  # Setting to -1 ensures the node itself is not selected
        similarity, sorted_indices = torch.sort(sim_scores_with_self_excluded, descending=False)# Ascending order sorting to get the least similar nodes first.
        neg_indices_mask = ~pos_mask[sorted_indices]
        selected_neg_indices = sorted_indices[neg_indices_mask][:k]
        '''
        
        
        max_sim = sim_matrix[i].max()
        sum_exp_pos_sim = torch.exp((sim_matrix[i, i] - max_sim) / tau) + torch.exp((pos_sim- max_sim) / tau).sum() 
        sum_exp_neg_sim = torch.exp((sim_matrix[i][selected_neg_indices] - max_sim) / tau).sum() 
        
        loss_val = -torch.log((sum_exp_pos_sim + epsilon) / (sum_exp_pos_sim + sum_exp_neg_sim + epsilon)) 
        contrastive_loss_values[i] = torch.where(torch.isfinite(loss_val), loss_val, torch.tensor(0.0, device=Hx.device))

    return contrastive_loss_values.mean()


'''
Positive: a node OR its neighbors OR two nodes NOT belong to the same HyE. 
Negative: else (selected top k neg nodes based on the furthest path/distance from the anchor node) OR (based on the least similar nodes)
'''
def combined_contrastive_loss_with_distance_based_neg_samples(Hx, Hy, hypergraph_dict1, hypergraph_dict2,edge_index, shortest_paths_tensor, cosine_similarity_matrix, tau=0.5, k=25):#0.5 default one
    num_nodes = Hx.size(0)
    sim_matrix = cosine_similarity(Hx, Hy)
    
    neighborhood_mask = create_neighborhood_mask(num_nodes, edge_index).to(Hx.device)
    hyperedge_mask1 = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=Hx.device)
    
    for hyperedge in hypergraph_dict1.values():
        for i in hyperedge:
            for j in hyperedge:
                if i!=j:
                    hyperedge_mask1[i, j] = 1
    '''
    hyperedge_mask2 = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=Hx.device)
    for hyperedge in hypergraph_dict2.values():
        for i in hyperedge:
            for j in hyperedge:
                if i!=j:
                    hyperedge_mask2[i, j] = 1
    '''
    '''
    hyperedge_mask2 = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=Hx.device)
    for node, ego_network in hypergraph_dict2.items():
        for neighbor in ego_network:
            if node != neighbor:
                hyperedge_mask2[node, neighbor] = 1
    '''
    
    temp1=~(hyperedge_mask1)
    combined_pos_mask = neighborhood_mask |temp1 
   
    contrastive_loss_values = torch.zeros(num_nodes, device=Hx.device)
    epsilon = 1e-8

    for i in range(num_nodes):
        pos_mask = combined_pos_mask[i]
        pos_sim = sim_matrix[i][pos_mask]

        '''
        path/distance-based
        '''
        
        distance_from_i = shortest_paths_tensor[i]
        distances, sorted_indices = torch.sort(distance_from_i, descending=True)# Descending order sorting to get the most distance nodes first.
        neg_indices_mask = ~pos_mask[sorted_indices]
        selected_neg_indices = sorted_indices[neg_indices_mask][:k]
        
        
        '''
        similarity-based 
        '''
        '''
        sim_scores_with_self_excluded = cosine_similarity_matrix[i].clone()
        sim_scores_with_self_excluded[i] = -1.0  # Setting to -1 ensures the node itself is not selected
        similarity, sorted_indices = torch.sort(sim_scores_with_self_excluded, descending=False)# Ascending order sorting to get the least similar nodes first.
        neg_indices_mask = ~pos_mask[sorted_indices]
        selected_neg_indices = sorted_indices[neg_indices_mask][:k]
        '''
        '''
        # Compute Log-Sum-Exp
        max_sim = sim_matrix[i].max()
        sum_exp_pos_sim = torch.exp((pos_sim - max_sim) / tau).sum()
        sum_exp_neg_sim = torch.exp((sim_matrix[i][selected_neg_indices] - max_sim) / tau) .sum()
        
        loss_val = -torch.log((sum_exp_pos_sim + epsilon) / (sum_exp_pos_sim + sum_exp_neg_sim + epsilon))
        contrastive_loss_values[i] = torch.where(torch.isfinite(loss_val), loss_val, torch.tensor(0.0, device=Hx.device))
        '''
        max_sim = sim_matrix[i].max()
        sum_exp_pos_sim = torch.exp((sim_matrix[i, i] - max_sim) / tau) + torch.exp((pos_sim- max_sim) / tau).sum() 
        sum_exp_neg_sim = torch.exp((sim_matrix[i][selected_neg_indices] - max_sim) / tau).sum() 
        
        loss_val = -torch.log((sum_exp_pos_sim + epsilon) / (sum_exp_pos_sim + sum_exp_neg_sim + epsilon)) 
        contrastive_loss_values[i] = torch.where(torch.isfinite(loss_val), loss_val, torch.tensor(0.0, device=Hx.device))
    return contrastive_loss_values.mean()
