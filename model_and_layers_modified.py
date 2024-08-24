import numpy as np
import torch
import torch.nn as nn
from dgl.nn import GraphConv
from data_preprocessing import DataProcessor
import networkx as nx
import torch.nn.functional as F
import copy
from config import device
import itertools
import bisect

# nl=np.eye(g.num_nodes())
# nl=torch.from_numpy(nl)
# nl=torch.tensor(nl,dtype=torch.long)


#GCN
'''
class GCN2(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN2, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, out_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
'''
class GCN3(nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GCN3, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, out_feats)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()        



#k-core
def k_core(graph):
    graph.remove_edges_from(nx.selfloop_edges(graph))
    k_core_dict = {node: 0 for node in graph.nodes()}

    degrees = dict(graph.degree())
    if not degrees:
        print("Graph has no nodes or edges.")
        return
    else:
        max_k = max(degrees.values())
        nodes_sorted_by_degree = sorted(degrees, key=degrees.get)

    for k in range(1, max_k + 1):
        while nodes_sorted_by_degree and degrees[nodes_sorted_by_degree[0]] < k:
            node = nodes_sorted_by_degree.pop(0)
            k_core_dict[node] = degrees[node]
            for neighbor in graph[node]:
                if degrees[neighbor] > degrees[node]:
                    degrees[neighbor] -= 1
                    position = nodes_sorted_by_degree.index(neighbor)
                    nodes_sorted_by_degree.pop(position)
                    bisect.insort(nodes_sorted_by_degree, neighbor, lo=position)
            del degrees[node]
            graph.remove_node(node)

    max_value = sum(k_core_dict.values())
    if max_value == 0:
        return k_core_dict
    else:
        normalized_dict = {key: value / max_value for key, value in k_core_dict.items()}
        return normalized_dict




#THTN1
class THTN1(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout):
        super(THTN1, self).__init__()

        self.query_dim = query_dim
        self.vtx_lin_1layer = torch.nn.Linear(input_dim, vertex_dim)
        self.vtx_lin = torch.nn.Linear(vertex_dim, vertex_dim)

        self.qe_lin = torch.nn.Linear(edge_dim, query_dim)
        self.kv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.vv_lin = torch.nn.Linear(vertex_dim, edge_dim)

        self.qv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.ke_lin = torch.nn.Linear(edge_dim, query_dim)
        self.ve_lin = torch.nn.Linear(edge_dim, vertex_dim)
        self.dropout = nn.Dropout(dropout)

        self.cls = nn.Linear(vertex_dim, number_class)
        self.mlp = nn.Linear(vertex_dim, 128)

        self.dropout = nn.Dropout(dropout)
        
        #add&norm
        self.layer_norm1 = nn.LayerNorm(vertex_dim)
        self.layer_norm2 = nn.LayerNorm(vertex_dim)

        #ffn
        self.linear1 = nn.Linear(vertex_dim, query_dim)
        self.linear3 = nn.Linear(vertex_dim, query_dim)
        self.relu = nn.ReLU()
        
        self.linear2 = nn.Linear(query_dim, vertex_dim)
        self.linear4 = nn.Linear(query_dim, vertex_dim)
    
    def add_and_norm1(self, x, residual):
        output = x + residual
        output = self.layer_norm1(output)
        return output

    def ffn1(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return output

    def add_and_norm2(self, x, residual):
        output = x + residual
        output = self.layer_norm2(output)
        return output

    def ffn2(self, x):
        output = self.linear3(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear4(output)
        return output
    def attention(self, edges):
        attn_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
        c=attn_score/np.sqrt(self.query_dim)
        return {'Attn': c}

    def message_func(self, edges):
        return {'v': edges.src['v'], 'Attn': edges.data['Attn']}

    def reduce_func1(self, nodes):
        attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
        return {'h': aggr}

    def reduce_func2(self, nodes):
        attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
        return {'h': aggr}

    def forward(self, hyG, vfeat, efeat, first_layer, last_layer):

        with hyG.local_scope():

            if first_layer:
                feat_v = self.vtx_lin_1layer(vfeat)
            else:
                feat_v = self.vtx_lin(vfeat)
            feat_e = efeat

            # node attention
            hyG.ndata['h'] = {'node': feat_v}
            hyG.ndata['k'] = {'node' : self.kv_lin(feat_v)}
            hyG.ndata['v'] = {'node' : self.vv_lin(feat_v)}
            hyG.ndata['q'] = {'edge' : self.qe_lin(feat_e)}
            hyG.apply_edges(self.attention, etype='in')
            hyG.update_all(self.message_func, self.reduce_func1, etype='in')
            feat_e_transformed = hyG.ndata['h']['edge']

            feat_e_add_norm=self.add_and_norm1(feat_e_transformed,feat_e)
            feat_e_ffn=self.ffn1(feat_e_add_norm)
            feat_e=self.add_and_norm1(feat_e_ffn,feat_e_add_norm)

            # edge attention
            hyG.ndata['k'] = {'edge' : self.ke_lin(feat_e)}
            hyG.ndata['v'] = {'edge' : self.ve_lin(feat_e)}
            hyG.ndata['q'] = {'node' : self.qv_lin(feat_v)}
            hyG.apply_edges(self.attention, etype='con')
            hyG.update_all(self.message_func, self.reduce_func2, etype='con')
            feat_v_transformed = hyG.ndata['h']['node']


            feat_v_add_norm=self.add_and_norm2(feat_v_transformed,feat_v)
            feat_v_ffn=self.ffn2(feat_v_add_norm)
            feat_v=self.add_and_norm2(feat_v_ffn,feat_v_add_norm)

            if not last_layer :
                feat_v = self.dropout(feat_v)
                return feat_v
            if last_layer:
                feat_v = self.dropout(feat_v)
                pred=self.cls(feat_v)
                feat_v=self.mlp(feat_v)
                
                return pred,feat_v
            else:
                return [g, feat_v, feat_e]


#Multi_head_thtn_1
class Multi_head_attn1(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout,num_heads,merge='cat'):
        super(Multi_head_attn1, self).__init__()

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(THTN1(input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout ))
        self.merge = merge

    def forward(self, hyG, v_feat, e_feat, first_layer, second_layer):
      # Get the outputs for each head and separate them into two lists
      head_outs_pred = []
      head_outs_feat = []
      for attn_head in self.heads:
          pred, feat = attn_head(hyG, v_feat, e_feat, first_layer, second_layer)
          head_outs_pred.append(pred)
          head_outs_feat.append(feat)

      if self.merge == 'cat':         
          return torch.cat(head_outs_pred, dim=1), torch.cat(head_outs_feat, dim=1) # concat on the output feature dimension (dim=1) for both pred and feat
      else:
          return torch.mean(torch.stack(head_outs_pred)), torch.mean(torch.stack(head_outs_feat)) # merge using average for both pred and feat

    def reset_parameters(self):
        self.heads.reset_parameters()
class THTN_attn1(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout, num_heads):
        super(THTN_attn1, self).__init__()
        self.layer1 = Multi_head_attn1(input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout, num_heads)

    def forward(self, hyG, v_feat, e_feat, first_layer, second_layer):
        x, y = self.layer1(hyG, v_feat, e_feat, first_layer, second_layer) # No changes here since we're already getting two return values from self.layer1
        return x, y


'''
#THTN2
class THTN2(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout):
        super(THTN2, self).__init__()

        self.query_dim = query_dim
        self.vtx_lin_1layer = torch.nn.Linear(input_dim, vertex_dim)
        self.vtx_lin = torch.nn.Linear(vertex_dim, vertex_dim)

        self.qe_lin = torch.nn.Linear(edge_dim, query_dim)
        self.kv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.vv_lin = torch.nn.Linear(vertex_dim, edge_dim)

        self.qv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.ke_lin = torch.nn.Linear(edge_dim, query_dim)
        self.ve_lin = torch.nn.Linear(edge_dim, vertex_dim)
        self.dropout = nn.Dropout(dropout)

        self.cls = nn.Linear(vertex_dim, number_class)
        self.mlp = nn.Linear(vertex_dim, 128)


    def attention(self, edges):
        attn_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
        c=attn_score/np.sqrt(self.query_dim)
        return {'Attn': c}

    def message_func(self, edges):
        return {'v': edges.src['v'], 'Attn': edges.data['Attn']}

    def reduce_func1(self, nodes):
        attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
        return {'h': aggr}

    def reduce_func2(self, nodes):
        attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)
        return {'h': aggr}

    def forward(self, hyG, vfeat, efeat, first_layer, last_layer):

        with hyG.local_scope():

            if first_layer:
                feat_v = self.vtx_lin_1layer(vfeat)
            else:
                feat_v = self.vtx_lin(vfeat)
            feat_e = efeat

            # node attention
            hyG.ndata['h'] = {'node': feat_v}
            hyG.ndata['k'] = {'node' : self.kv_lin(feat_v)}
            hyG.ndata['v'] = {'node' : self.vv_lin(feat_v)}
            hyG.ndata['q'] = {'edge' : self.qe_lin(feat_e)}
            hyG.apply_edges(self.attention, etype='in')
            hyG.update_all(self.message_func, self.reduce_func1, etype='in')
            feat_e = hyG.ndata['h']['edge']

            # edge attention
            hyG.ndata['k'] = {'edge' : self.ke_lin(feat_e)}
            hyG.ndata['v'] = {'edge' : self.ve_lin(feat_e)}
            hyG.ndata['q'] = {'node' : self.qv_lin(feat_v)}
            hyG.apply_edges(self.attention, etype='con')
            hyG.update_all(self.message_func, self.reduce_func2, etype='con')
            feat_v = hyG.ndata['h']['node']

            if not last_layer :
                feat_v = F.dropout(feat_v, self.dropout)
                return feat_v
            if last_layer:
                feat_v = self.dropout(feat_v)
                pred=self.cls(feat_v)
                feat_v=self.mlp(feat_v)
                
                return pred,feat_v
            else:
                return [g, feat_v, feat_e]

#Multi_head_thtn_2
class Multi_head_attn2(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout,num_heads,merge='cat'):
        super(Multi_head_attn2, self).__init__()

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(THTN2(input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout ))
        self.merge = merge

    def forward(self, hyG, v_feat, e_feat, first_layer, second_layer):
      # Get the outputs for each head and separate them into two lists
      head_outs_pred = []
      head_outs_feat = []
      for attn_head in self.heads:
          pred, feat = attn_head(hyG, v_feat, e_feat, first_layer, second_layer)
          head_outs_pred.append(pred)
          head_outs_feat.append(feat)

      if self.merge == 'cat':
          return torch.cat(head_outs_pred, dim=1), torch.cat(head_outs_feat, dim=1) # concat on the output feature dimension (dim=1) for both pred and feat
      else:
          # merge using average for both pred and feat
          return torch.mean(torch.stack(head_outs_pred)), torch.mean(torch.stack(head_outs_feat))

    def reset_parameters(self):
        self.heads.reset_parameters()
class THTN_attn2(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout, num_heads):
        super(THTN_attn2, self).__init__()
        self.layer1 = Multi_head_attn2(input_dim, query_dim, vertex_dim, edge_dim, number_class, dropout, num_heads)

    def forward(self, hyG, v_feat, e_feat, first_layer, second_layer):       
        x, y = self.layer1(hyG, v_feat, e_feat, first_layer, second_layer) # No changes here since we're already getting two return values from self.layer1
        return x, y
'''


#Structure-aware THTN
def centrality_values_function(G):
    centrality_values=list(nx.closeness_centrality(G).values())
    centrality_values = torch.LongTensor(centrality_values).to(device)
    return centrality_values

def local_clustering_coefficient(com_graph):
    clustering_coefficients = {}
    hyperedge_nodes = list(com_graph.nodes())
    for node in hyperedge_nodes:
        neighbors_in_hyperedge = set(com_graph.neighbors(node))
        n = len(neighbors_in_hyperedge)
        if n > 1:
            # Calculate the number of existing connections between neighbors
            existing_connections = sum(1 for u, v in itertools.combinations(neighbors_in_hyperedge, 2) if com_graph.has_edge(u, v))
            total_possible_connections = n * (n - 1) // 2
            clustering_coefficients[node] = existing_connections / total_possible_connections
        else:
            clustering_coefficients[node] = 0

    return clustering_coefficients


def hyperedge_clustering_coefficient(com_graph):
    hyperedge_nodes = list(com_graph.nodes())
    triangle_count = 0

    for idx, u in enumerate(hyperedge_nodes):
        for v in hyperedge_nodes[idx + 1:]:
            if com_graph.has_edge(u, v):  # Ensure there is an edge between u and v
                # common_neighbors = list(nx.common_neighbors(com_graph, u, v))
                # for w in common_neighbors:
                #     if com_graph.has_edge(u, w) and com_graph.has_edge(v, w):  # Check if u, v, w form a triangle
                        triangle_count += len(list(nx.common_neighbors(com_graph, u, v)))
                        # triangle_count += 1

    # Since each triangle is counted three times, divide by 3
    triangle_count //= 3
    n = len(hyperedge_nodes)
    total_possible_triangles = n * (n - 1) * (n - 2) // 6

    clustering_coefficient = triangle_count / total_possible_triangles if total_possible_triangles > 0 else 0.0
    return clustering_coefficient




class Structure_aware_THTN2(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class,centrality_values,uniqueness,dropout,LEN,num_heads,G):
        super(Structure_aware_THTN2, self).__init__()

        self.G=G
        self.query_dim = query_dim
        self.vtx_lin_1layer = torch.nn.Linear(input_dim, vertex_dim)
        self.vtx_lin = torch.nn.Linear(vertex_dim, vertex_dim)

        self.qe_lin = torch.nn.Linear(edge_dim, query_dim)
        self.kv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.vv_lin = torch.nn.Linear(vertex_dim, edge_dim)

        self.qv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.ke_lin = torch.nn.Linear(edge_dim, query_dim)
        self.ve_lin = torch.nn.Linear(edge_dim, vertex_dim)

        self.cls = nn.Linear(vertex_dim, number_class)
        self.mlp = nn.Linear(vertex_dim, 128)

        self.cs_embedding = nn.Embedding(num_embeddings=len(centrality_values), embedding_dim=LEN)
        self.un_embedding = nn.Embedding(num_embeddings=len(uniqueness), embedding_dim=LEN)

        #self.eign_vec_lin=torch.nn.Linear(eign_vec.shape[1],vertex_dim)
        self.dropout = nn.Dropout(dropout)
        
        #add&norm
        self.layer_norm1 = nn.LayerNorm(vertex_dim)
        self.layer_norm2 = nn.LayerNorm(vertex_dim)

        #ffn
        self.linear1 = nn.Linear(vertex_dim, query_dim)
        self.linear3 = nn.Linear(vertex_dim, query_dim)
        self.relu = nn.ReLU()
        
        self.linear2 = nn.Linear(query_dim, vertex_dim)
        self.linear4 = nn.Linear(query_dim, vertex_dim)


    def set_graph_data(self, egos_G_refined, DICT_refined_natural):
        self.egos_G_refined = egos_G_refined
        self.DICT_refined_natural = DICT_refined_natural
    
    def add_and_norm1(self, x, residual):
        output = x + residual
        output = self.layer_norm1(output)
        return output

    def ffn1(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return output

    def add_and_norm2(self, x, residual):
        output = x + residual
        output = self.layer_norm2(output)
        return output

    def ffn2(self, x):
        output = self.linear3(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear4(output)
        return output

    def attention(self, edges):
        attn_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
        c=attn_score/np.sqrt(self.query_dim)
        return {'Attn': c}

    def message_func(self, edges):
        return {'v': edges.src['v'], 'Attn': edges.data['Attn']}

    def reduce_func1(self, nodes):
        #attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        #aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)


        lst_node=nodes.nodes().tolist()
        updated_attention_score_list=[]
        node_embedding_gcn_list=[]
        for i in range (len(lst_node)):
          node=lst_node[i]
          attn_score=nodes.mailbox['Attn'][i]

          # Calculate clustering coefficients for nodes within the hyperedge
          com_graph=nx.Graph()
          com_graph=copy.deepcopy(self.egos_G_refined[node])

          clustering_coefficients = local_clustering_coefficient(self.egos_G_refined[node])
          clustering_coefficients=torch.Tensor(list(clustering_coefficients.values())).to(device)

          '''  
          k_core_values=list(k_core(com_graph).values())
          k_core_values=torch.Tensor(k_core_values).to(device)
          '''
          k_core_values=0  
          combined_centrality_values=k_core_values+clustering_coefficients

          updated_attention_score=torch.add(attn_score,combined_centrality_values)
          updated_attention_score=updated_attention_score.tolist()
          updated_attention_score_list.append(updated_attention_score)

        updated_attention_score_list=torch.Tensor(updated_attention_score_list).to(device)


        updated_attention_score_list = F.softmax(updated_attention_score_list, dim=1)
        aggr = torch.sum(updated_attention_score_list.unsqueeze(-1) *nodes.mailbox['v'], dim=1)

        return {'h': aggr}

    def reduce_func2(self, nodes):
        lst_node=nodes.nodes().tolist()

        #attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        #aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)

        updated_attention_score_list=[]
        for i in range (len(lst_node)):
          node=lst_node[i]
          attn_score=nodes.mailbox['Attn'][i]

          key_=[key for key,value in self.DICT_refined_natural.items() if node in value]
          betweenness_centrality_values=[]

          for k in key_:

            #c=hyperedge_clustering_coefficient(coms_G[k])
            #c=hyperedge_clustering_coefficient(self.coms_G_refined[k])
            c=0
            d=c+(self.egos_G_refined[k].number_of_nodes()/self.G.number_of_nodes())
            betweenness_centrality_values.append(d)

          betweenness_centrality_values=torch.Tensor(betweenness_centrality_values).to(device)
          updated_attention_score=torch.add(attn_score,betweenness_centrality_values)
          updated_attention_score=updated_attention_score.tolist()
          updated_attention_score_list.append(updated_attention_score)

        updated_attention_score_list=torch.Tensor(updated_attention_score_list).to(device)
        updated_attention_score_list = F.softmax(updated_attention_score_list, dim=1)

        aggr = torch.sum(updated_attention_score_list.unsqueeze(-1) * nodes.mailbox['v'], dim=1)


        return {'h': aggr}

    def forward(self, g,hyG, centrality_values, uniqueness,gcn_model,node_feat,egos_G_refined,DICT_refined_natural,vfeat, efeat, first_layer, last_layer):

        with hyG.local_scope():

            if first_layer:
                feat_v = self.vtx_lin_1layer(vfeat)
                #pe=self.eign_vec_lin(eign_vec)
                cs=self.cs_embedding(centrality_values)
                un=self.un_embedding(uniqueness)


                #feat_v=feat_v+pe
                #feat_v_gcn=gcn_model(g,node_feat).to(device)
                #feat_v=feat_v+feat_v_gcn
                feat_v=feat_v+cs
                feat_v=feat_v+un
            else:
                feat_v = self.vtx_lin(vfeat)
                #pe=self.eign_vec_lin(eign_vec)
                cs=self.cs_embedding(centrality_values)
                un=self.un_embedding(uniqueness)


                #feat_v=feat_v+pe
                #feat_v_gcn=gcn_model(g,node_feat).to(device)
                #feat_v=feat_v+feat_v_gcn
                feat_v=feat_v+cs
                feat_v=feat_v+un

            feat_e = efeat

            # node attention
            hyG.ndata['h'] = {'node': feat_v}
            hyG.ndata['k'] = {'node' : self.kv_lin(feat_v)}
            hyG.ndata['v'] = {'node' : self.vv_lin(feat_v)}
            hyG.ndata['q'] = {'edge' : self.qe_lin(feat_e)}
            hyG.apply_edges(self.attention, etype='in')
            hyG.update_all(self.message_func, self.reduce_func1, etype='in')
            feat_e_transformed = hyG.ndata['h']['edge']

            feat_e_add_norm=self.add_and_norm1(feat_e_transformed,feat_e)
            feat_e_ffn=self.ffn1(feat_e_add_norm)
            feat_e=self.add_and_norm1(feat_e_ffn,feat_e_add_norm)

            # edge attention
            hyG.ndata['k'] = {'edge' : self.ke_lin(feat_e)}
            hyG.ndata['v'] = {'edge' : self.ve_lin(feat_e)}
            hyG.ndata['q'] = {'node' : self.qv_lin(feat_v)}
            hyG.apply_edges(self.attention, etype='con')
            hyG.update_all(self.message_func, self.reduce_func2, etype='con')
            feat_v_transformed = hyG.ndata['h']['node']

            feat_v_add_norm=self.add_and_norm2(feat_v_transformed,feat_v)
            feat_v_ffn=self.ffn2(feat_v_add_norm)
            feat_v=self.add_and_norm2(feat_v_ffn,feat_v_add_norm)

            if not last_layer :
                feat_v = self.dropout(feat_v)
                pred=0
                feat_v=self.mlp(feat_v)
                return pred,feat_v
            if last_layer:
                feat_v = self.dropout(feat_v)
                pred=self.cls(feat_v)
                feat_v=self.mlp(feat_v)
                return pred,feat_v
            else:
                return [g, feat_v, feat_e]


#Multi_head_thtn_structure_aware
class Multi_head_attn_structure_aware2(nn.Module):

    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim,                                                                                                     number_class,centrality_values,uniqueness,dropout,LEN,num_heads,G,merge='cat'):
        super(Multi_head_attn_structure_aware2, self).__init__()

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(Structure_aware_THTN2(input_dim, query_dim, vertex_dim, edge_dim, number_class,                                                               centrality_values,uniqueness,dropout,LEN,num_heads,G ))
        self.merge = merge

    def forward(self, g,hyG, centrality_values, uniqueness,gcn_model,node_feat, egos_G_refined,DICT_refined_natural,v_feat, e_feat, first_layer,second_layer):
      for head in self.heads:
          head.set_graph_data(egos_G_refined, DICT_refined_natural)
          
      # Get the outputs for each head and separate them into two lists
      head_outs_pred = []
      head_outs_feat = []
      for attn_head in self.heads:
          pred, feat = attn_head(g,hyG, centrality_values,  uniqueness,gcn_model,node_feat, egos_G_refined,DICT_refined_natural,v_feat, e_feat, first_layer,           second_layer)
          head_outs_pred.append(pred)
          head_outs_feat.append(feat)

      if self.merge == 'cat':
          return torch.cat(head_outs_pred, dim=1), torch.cat(head_outs_feat, dim=1) # concat on the output feature dimension (dim=1) for both pred and feat
      else:
          return torch.mean(torch.stack(head_outs_pred)), torch.mean(torch.stack(head_outs_feat)) # merge using average for both pred and feat

    def reset_parameters(self):
        self.heads.reset_parameters()
class Structure_aware_THTN_attn2(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class,centrality_values,uniqueness,dropout,LEN, num_heads,G):
        super(Structure_aware_THTN_attn2, self).__init__()
        
        self.layer1 = Multi_head_attn_structure_aware2(input_dim, query_dim, vertex_dim, edge_dim, number_class,centrality_values,uniqueness,                  dropout,LEN, num_heads,G)

    def forward(self,g,hyG, centrality_values,  uniqueness,gcn_model, node_feat,egos_G_refined,DICT_refined_natural,v_feat, e_feat, first_layer,                      second_layer):
        x, y = self.layer1(g,hyG, centrality_values,  uniqueness,gcn_model,node_feat, egos_G_refined,DICT_refined_natural,v_feat, e_feat, first_layer,               second_layer) # No changes here since we're already getting two return values from self.layer1
        return x, y





class Structure_aware_THTN3(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class,centrality_values,uniqueness,dropout,LEN,num_heads,G):
        super(Structure_aware_THTN3, self).__init__()

        self.G=G
        self.query_dim = query_dim
        self.vtx_lin_1layer = torch.nn.Linear(input_dim, vertex_dim)
        self.vtx_lin = torch.nn.Linear(vertex_dim, vertex_dim)

        self.qe_lin = torch.nn.Linear(edge_dim, query_dim)
        self.kv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.vv_lin = torch.nn.Linear(vertex_dim, edge_dim)

        self.qv_lin = torch.nn.Linear(vertex_dim, query_dim)
        self.ke_lin = torch.nn.Linear(edge_dim, query_dim)
        self.ve_lin = torch.nn.Linear(edge_dim, vertex_dim)

        self.cls = nn.Linear(vertex_dim, number_class)
        self.mlp = nn.Linear(vertex_dim, 128)

        self.cs_embedding = nn.Embedding(num_embeddings=len(centrality_values), embedding_dim=LEN)
        self.un_embedding = nn.Embedding(num_embeddings=len(uniqueness), embedding_dim=LEN)

        #self.eign_vec_lin=torch.nn.Linear(eign_vec.shape[1],vertex_dim)

        #add&norm
        self.layer_norm1 = nn.LayerNorm(vertex_dim)
        self.layer_norm2 = nn.LayerNorm(vertex_dim)

        #ffn
        self.linear1 = nn.Linear(vertex_dim, query_dim)
        self.linear3 = nn.Linear(vertex_dim, query_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(query_dim, vertex_dim)
        self.linear4 = nn.Linear(query_dim, vertex_dim)


    def set_graph_data(self, coms_G_refined, DICT_refined):
        self.coms_G_refined = coms_G_refined
        self.DICT_refined = DICT_refined
    
    def add_and_norm1(self, x, residual):
        output = x + residual
        output = self.layer_norm1(output)
        return output

    def ffn1(self, x):
        output = self.linear1(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear2(output)
        return output

    def add_and_norm2(self, x, residual):
        output = x + residual
        output = self.layer_norm2(output)
        return output

    def ffn2(self, x):
        output = self.linear3(x)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.linear4(output)
        return output

    def attention(self, edges):
        attn_score = F.leaky_relu((edges.src['k'] * edges.dst['q']).sum(-1))
        c=attn_score/np.sqrt(self.query_dim)
        return {'Attn': c}

    def message_func(self, edges):
        return {'v': edges.src['v'], 'Attn': edges.data['Attn']}

    def reduce_func1(self, nodes):
        #attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        #aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)


        lst_node=nodes.nodes().tolist()
        updated_attention_score_list=[]
        node_embedding_gcn_list=[]
        for i in range (len(lst_node)):
          node=lst_node[i]
          attn_score=nodes.mailbox['Attn'][i]

          # Calculate clustering coefficients for nodes within the hyperedge
          com_graph=nx.Graph()
          com_graph=copy.deepcopy(self.coms_G_refined[node])

          clustering_coefficients = local_clustering_coefficient(self.coms_G_refined[node])
          clustering_coefficients=torch.Tensor(list(clustering_coefficients.values())).to(device)

          '''  
          k_core_values=list(k_core(com_graph).values())
          k_core_values=torch.Tensor(k_core_values).to(device)
          '''
          k_core_values=0  
          combined_centrality_values=k_core_values+clustering_coefficients

          updated_attention_score=torch.add(attn_score,combined_centrality_values)
          updated_attention_score=updated_attention_score.tolist()
          updated_attention_score_list.append(updated_attention_score)

        updated_attention_score_list=torch.Tensor(updated_attention_score_list).to(device)


        updated_attention_score_list = F.softmax(updated_attention_score_list, dim=1)
        aggr = torch.sum(updated_attention_score_list.unsqueeze(-1) *nodes.mailbox['v'], dim=1)

        return {'h': aggr}

    def reduce_func2(self, nodes):
        lst_node=nodes.nodes().tolist()

        #attention_score = F.softmax((nodes.mailbox['Attn']), dim=1)
        #aggr = torch.sum(attention_score.unsqueeze(-1) * nodes.mailbox['v'], dim=1)

        updated_attention_score_list=[]
        for i in range (len(lst_node)):
          node=lst_node[i]
          attn_score=nodes.mailbox['Attn'][i]

          key_=[key for key,value in self.DICT_refined.items() if node in value]
          betweenness_centrality_values=[]

          for k in key_:

            #c=hyperedge_clustering_coefficient(coms_G[k])
            #c=hyperedge_clustering_coefficient(self.coms_G_refined[k])
            c=0
            d=c+(self.coms_G_refined[k].number_of_nodes()/self.G.number_of_nodes())
            betweenness_centrality_values.append(d)

          betweenness_centrality_values=torch.Tensor(betweenness_centrality_values).to(device)
          updated_attention_score=torch.add(attn_score,betweenness_centrality_values)
          updated_attention_score=updated_attention_score.tolist()
          updated_attention_score_list.append(updated_attention_score)

        updated_attention_score_list=torch.Tensor(updated_attention_score_list).to(device)
        updated_attention_score_list = F.softmax(updated_attention_score_list, dim=1)

        aggr = torch.sum(updated_attention_score_list.unsqueeze(-1) * nodes.mailbox['v'], dim=1)


        return {'h': aggr}

    def forward(self, g,hyG, centrality_values,uniqueness, gcn_model,node_feat,coms_G_refined,DICT_refined,vfeat, efeat, first_layer, last_layer):

        with hyG.local_scope():

            if first_layer:
                feat_v = self.vtx_lin_1layer(vfeat)
                #pe=self.eign_vec_lin(eign_vec)
                cs=self.cs_embedding(centrality_values)
                un=self.un_embedding(uniqueness)


                #feat_v=feat_v+pe
                feat_v_gcn=gcn_model(g,node_feat).to(device)
                feat_v=feat_v+feat_v_gcn
                feat_v=feat_v+cs
                feat_v=feat_v+un
            else:
                feat_v = self.vtx_lin(vfeat)
                #pe=self.eign_vec_lin(eign_vec)
                cs=self.cs_embedding(centrality_values)
                un=self.un_embedding(uniqueness)


                #feat_v=feat_v+pe
                feat_v_gcn=gcn_model(g,node_feat).to(device)
                feat_v=feat_v+feat_v_gcn
                feat_v=feat_v+cs
                feat_v=feat_v+un

            feat_e = efeat

            # node attention
            hyG.ndata['h'] = {'node': feat_v}
            hyG.ndata['k'] = {'node' : self.kv_lin(feat_v)}
            hyG.ndata['v'] = {'node' : self.vv_lin(feat_v)}
            hyG.ndata['q'] = {'edge' : self.qe_lin(feat_e)}
            hyG.apply_edges(self.attention, etype='in')
            hyG.update_all(self.message_func, self.reduce_func1, etype='in')
            feat_e_transformed = hyG.ndata['h']['edge']

            feat_e_add_norm=self.add_and_norm1(feat_e_transformed,feat_e)
            feat_e_ffn=self.ffn1(feat_e_add_norm)
            feat_e=self.add_and_norm1(feat_e_ffn,feat_e_add_norm)

            # edge attention
            hyG.ndata['k'] = {'edge' : self.ke_lin(feat_e)}
            hyG.ndata['v'] = {'edge' : self.ve_lin(feat_e)}
            hyG.ndata['q'] = {'node' : self.qv_lin(feat_v)}
            hyG.apply_edges(self.attention, etype='con')
            hyG.update_all(self.message_func, self.reduce_func2, etype='con')
            feat_v_transformed = hyG.ndata['h']['node']


            feat_v_add_norm=self.add_and_norm2(feat_v_transformed,feat_v)
            feat_v_ffn=self.ffn2(feat_v_add_norm)
            feat_v=self.add_and_norm2(feat_v_ffn,feat_v_add_norm)

            if not last_layer :
                feat_v = self.dropout(feat_v)
                pred=0
                feat_v=self.mlp(feat_v)
                return pred,feat_v
            if last_layer:
                feat_v = self.dropout(feat_v)
                pred=self.cls(feat_v)
                feat_v=self.mlp(feat_v)
                return pred,feat_v
            else:
                return [g, feat_v, feat_e]


#Multi_head_thtn_structure_aware
class Multi_head_attn_structure_aware3(nn.Module):

    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim,                                                                                                     number_class,centrality_values,uniqueness,dropout,LEN,num_heads,G,merge='cat'):
        super(Multi_head_attn_structure_aware3, self).__init__()

        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(Structure_aware_THTN3(input_dim, query_dim, vertex_dim, edge_dim, number_class,                                                               centrality_values,uniqueness,dropout,LEN,num_heads,G ))
        self.merge = merge

    def forward(self, g,hyG, centrality_values,uniqueness,gcn_model,node_feat,coms_G_refined,DICT_refined,v_feat, e_feat, first_layer,                   second_layer):
      
      for head in self.heads:
          head.set_graph_data(coms_G_refined, DICT_refined)
          
      # Get the outputs for each head and separate them into two lists
      head_outs_pred = []
      head_outs_feat = []
      for attn_head in self.heads:
          pred, feat = attn_head(g,hyG,centrality_values, uniqueness, gcn_model,node_feat, coms_G_refined,DICT_refined,v_feat, e_feat, first_layer,           second_layer)
          head_outs_pred.append(pred)
          head_outs_feat.append(feat)

      if self.merge == 'cat':
          return torch.cat(head_outs_pred, dim=1), torch.cat(head_outs_feat, dim=1) # concat on the output feature dimension (dim=1) for both pred and feat
      else:
          return torch.mean(torch.stack(head_outs_pred)), torch.mean(torch.stack(head_outs_feat)) # merge using average for both pred and feat

    def reset_parameters(self):
        self.heads.reset_parameters()
class Structure_aware_THTN_attn3(nn.Module):
    def __init__(self, input_dim, query_dim, vertex_dim, edge_dim, number_class,centrality_values,uniqueness,dropout,LEN,num_heads,G):
        super(Structure_aware_THTN_attn3, self).__init__()
        
        self.layer1 = Multi_head_attn_structure_aware3(input_dim, query_dim, vertex_dim, edge_dim, number_class,centrality_values,uniqueness,                  dropout,LEN,num_heads,G)

    def forward(self,g,hyG, centrality_values,uniqueness,gcn_model, node_feat,coms_G_refined,DICT_refined,v_feat, e_feat, first_layer,                      second_layer):
        
        x, y = self.layer1(g,hyG, centrality_values, uniqueness, gcn_model,node_feat, coms_G_refined,DICT_refined,v_feat, e_feat, first_layer,               second_layer) # No changes here since we're already getting two return values from self.layer1
        return x, y