import pickle
import random
import time
import torch.nn.functional as F
import igraph
import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_max_pool

from utils.util1 import laplacian_positional_encoding, get_graph_representation


class GraphGenerator(nn.Module):
    training = True
    weights = None

    def __init__(self, args):
        super().__init__()
        self.dropout = 0.5
        self.max_count = 66
        self.latent_size = 64
        self.device = torch.device(args.device)
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        with open(args.ontology_path, "rb") as f:
            ontology_embeddings = pickle.load(f)
        self.ontology, self.type2id, self.type_embeddings = ontology_embeddings
        self.type_list = [k for k in self.ontology["events"]]
        self.type_list.append("start")
        self.type_list.append("end")
        self.type_count = len(self.type_list)
        self.ins_init = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.LayerNorm([self.hidden_size], elementwise_affine=False)
        )
        self.sch_init = nn.Sequential(
            nn.Linear(self.emb_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.LayerNorm([self.hidden_size], elementwise_affine=False)
        )
        self.Transformer = Transformer(args)
        self.LayerNorm1 = nn.LayerNorm([self.hidden_size], elementwise_affine=False)
        self.MLP_mean = nn.Sequential(
            nn.Linear(self.hidden_size * 1, self.hidden_size // 2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.latent_size),
            nn.Dropout(self.dropout),
            nn.LayerNorm([self.latent_size], elementwise_affine=False)
        )
        self.MLP_logvar = nn.Sequential(
            nn.Linear(self.hidden_size * 1, self.hidden_size // 2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size // 2, self.latent_size),
            nn.Dropout(self.dropout),
            nn.LayerNorm([self.latent_size], elementwise_affine=False)
        )
        self.map_e = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.LayerNorm([self.hidden_size], elementwise_affine=False)
        )
        self.grud = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.attention = nn.MultiheadAttention(self.hidden_size, 8, dropout=self.dropout, batch_first=True)
        self.define_type = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.type_count),
        )
        self.add_edge = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
            nn.Sigmoid(),
        )
        self.loss_function1 = MultiClassFocalLossWithAlpha(
            alpha=[1 / self.type_count] * self.type_count, device=args.device
        )
        self.loss_function2 = nn.BCEWithLogitsLoss()

    def forward(self, g_batch, stochastic=True):
        L1, L2 = 0, 0
        self.init_graph(g_batch)
        G, H_global = self.Transformer(g_batch)
        mean, logvar = self.MLP_mean(H_global), self.MLP_logvar(H_global)
        kld = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        z = self.sample(mean, logvar)
        H_z = self.map_e(z)
        # train stage: each node count
        # create new graphs
        G_re = [igraph.Graph(directed=True) for _ in G]
        for step, g in enumerate(G):
            h_global = H_z[step: step + 1]
            # set ground truth
            node_truth = [self.type_list.index(v["type"]) for v in g.vs]
            node_truth = torch.tensor(node_truth, dtype=torch.int64, device=self.device)
            # 换成时序增强图
            enhanced_edges = g["enhanced_edges"]
            enhanced_adj = self.create_adjacency_matrix(enhanced_edges, g.vcount())
            edge_truth = torch.tensor(enhanced_adj, dtype=torch.float32, device=self.device)
            edge_truth = edge_truth.view(-1, 2)
            H_in = h_global.unsqueeze(1).expand(-1, g.vcount(), -1)
            H_out, _ = self.grud(H_in)
            attn_mask = torch.triu(torch.ones(g.vcount(), g.vcount()), diagonal=1).to(dtype=torch.bool)
            H_attn, _ = self.attention(H_out, H_out, H_out, attn_mask=attn_mask.to(self.device))
            H_out = self.LayerNorm1(H_out + H_attn)
            H_out = H_out.squeeze(0)
            # define node types
            type_probs = torch.softmax(self.define_type(H_out), dim=-1)
            L1 = L1 + self.loss_function1(type_probs, node_truth)
            type_probs_numpy = type_probs.cpu().detach().numpy()
            # random select type
            g_re = G_re[step]
            g_re.add_vertices(g.vcount())
            for step2, v in enumerate(g.vs):
                if stochastic:
                    new_type = np.random.choice(range(self.type_count), p=type_probs_numpy[step2])
                else:
                    new_type = torch.argmax(type_probs[step2], 0).item()
                g_re.vs[step2]["type"] = self.type_list[new_type]
                g_re.vs[step2]["H"] = H_out[step2].view(1, -1)
            edges = [torch.cat([g_re.vs[ii]["H"], g_re.vs[jj]["H"]], dim=-1)
                     for ii in range(g_re.vcount()) for jj in range(g_re.vcount())]
            H_edges = torch.cat(edges, dim=0)
            edge_probs = torch.softmax(self.add_edge(H_edges), dim=-1)
            L2 = L2 + self.loss_function2(edge_probs, edge_truth)
            edge_probs_numpy = edge_probs.cpu().detach().numpy()
            i = 0
            for ii in range(g_re.vcount()):
                for jj in range(g_re.vcount()):
                    if stochastic:
                        edge_exist = np.random.choice([0, 1], p=edge_probs_numpy[i])
                    else:
                        edge_exist = torch.argmax(edge_probs[i], 0).item()
                    if edge_exist:
                        g_re.add_edge(ii, jj)
                    i += 1
        L = L1 + L2 + kld
        return G_re, L.float()

    def forward_test(self, g_batch, stochastic=True):
        self.init_graph(g_batch)
        G, H_global = self.Transformer(g_batch)
        mean, logvar = self.MLP_mean(H_global), self.MLP_logvar(H_global)
        z = self.sample(mean, logvar)
        H_z = self.map_e(z)
        # test stage: max count
        H_in = H_z.unsqueeze(1).expand(-1, self.max_count, -1)
        H_out, _ = self.grud(H_in)
        attn_mask = torch.triu(torch.ones(self.max_count, self.max_count), diagonal=1).to(dtype=torch.bool)
        H_attn, _ = self.attention(H_out, H_out, H_out, attn_mask=attn_mask.to(self.device))
        H_out = self.LayerNorm1(H_out + H_attn)
        # define node types
        type_probs = torch.softmax(self.define_type(H_out), dim=-1)
        type_probs_numpy = type_probs.cpu().detach().numpy()
        # create new graphs
        G_re = [igraph.Graph(directed=True) for _ in G]
        # random select type
        for step1, g in enumerate(G_re):
            g.add_vertices(self.max_count)
            for step2, v in enumerate(g.vs):
                if stochastic:
                    new_type = np.random.choice(range(self.type_count), p=type_probs_numpy[step1, step2])
                else:
                    new_type = torch.argmax(type_probs[step1, step2], 0).item()
                G_re[step1].vs[step2]["type"] = self.type_list[new_type]
                G_re[step1].vs[step2]["H"] = H_out[step1, step2].view(1, -1)
        # add edges according to node embeddings
        edges = [torch.cat([g_re.vs[ii]["H"], g_re.vs[jj]["H"]], dim=-1) for g_re in G_re
                 for ii in range(g_re.vcount()) for jj in range(g_re.vcount())]
        H_edges = torch.cat(edges, dim=0)
        edge_probs = torch.softmax(self.add_edge(H_edges), dim=-1)
        edge_probs_numpy = edge_probs.cpu().detach().numpy()
        i = 0
        for step1, g_re in enumerate(G_re):
            for ii in range(g_re.vcount()):
                for jj in range(g_re.vcount()):
                    if stochastic:
                        edge_exist = np.random.choice([0, 1], p=edge_probs_numpy[i])
                    else:
                        edge_exist = torch.argmax(edge_probs[i], 0).item()
                    if edge_exist:
                        G_re[step1].add_edge(ii, jj)
                    i += 1
        return G_re, torch.tensor(0.)

    def create_adjacency_matrix(self, edges, max_node):
        # 初始化邻接矩阵
        adj_matrix = [[[1.0, 0.0] for _ in range(max_node)] for _ in range(max_node)]

        # 遍历每条边，填充邻接矩阵
        for edge in edges:
            i, j, p = edge
            adj_matrix[i][j] = [1 - p, p]

        return adj_matrix

    def sample(self, mu, logvar, eps_scale=0.01):
        # return z ~ N(mu, std)
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.randn_like(std) * eps_scale
            return eps.mul(std).add_(mu)
        else:
            return mu

    def init_graph(self, g_batch):
        for i, g in enumerate(g_batch):
            for v in range(g.vcount()):
                H = self.sch_init(g.vs[v]["oH"].to(self.device))
                g_batch[i].vs[v]["H"] = H

    def get_p_emb(self, provenances, p_embedding, p2id):
        provens = self.remove_duplication(provenances)
        proven_embeddings = [p_embedding[p2id[p]].view(1, -1) for p in provens]
        proven_embeddings = torch.cat(proven_embeddings, dim=0)
        proven_embeddings = torch.mean(proven_embeddings, dim=0, keepdim=True)
        return proven_embeddings

    def remove_duplication(self, L):
        L_clear = []
        for p in L:
            if p not in L_clear:
                L_clear.append(p)
        return L_clear


class Transformer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = 0.5
        self.L = args.L
        self.R = args.R
        self.en = args.encoder_count
        self.dn = args.decoder_count
        self.pos_enc_dim = args.pos_enc_dim
        self.device = torch.device(args.device)
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.embedding_lap_pos_enc = nn.Linear(self.pos_enc_dim, self.hidden_size)
        self.Encoder1s = nn.ModuleList([Encoder1(args) for _ in range(self.en)])
        self.Encoder2s = nn.ModuleList([Encoder2(args) for _ in range(self.dn)])

    def forward(self, G):
        # Encoder1
        if self.L:
            for g in G:
                # 拉普拉斯特征向量
                Laplacian = laplacian_positional_encoding(g, self.pos_enc_dim).to(self.device)
                h_lap_pos_enc = self.embedding_lap_pos_enc(Laplacian)
                h = torch.cat([v["H"] for v in g.vs], dim=0)
                h = (h + h_lap_pos_enc).unsqueeze(0)
                for encoder in self.Encoder1s:
                    h = encoder(h)
                # add embedding to nodes
                h = h.squeeze(0)
                for vi in range(g.vcount()):
                    g.vs[vi]["H"] = h[vi: vi + 1]
        # Encoder2
        x, edge_index, batch = get_graph_representation(G, self.device)
        if self.R:
            for encoder in self.Encoder2s:
                x = encoder(x, edge_index, batch)
            i = 0
            for ii in range(len(G)):
                for jj in range(G[ii].vcount()):
                    G[ii].vs[jj]["H"] = x[i: i + 1]
                    i += 1
        h_global = global_max_pool(x, batch)
        # Decoder ==> create node list
        # RNN
        return G, h_global


class Encoder1(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = 0.5
        self.device = torch.device(args.device)
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.attention1 = nn.MultiheadAttention(self.hidden_size, 8, dropout=self.dropout, batch_first=True)
        self.LayerNorm1 = nn.LayerNorm([self.hidden_size], elementwise_affine=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

    def forward(self, h):
        attn_output, attn_weight = self.attention1(h, h, h)
        h = self.LayerNorm1(h + attn_output)
        h1 = self.feed_forward(h)
        h = self.LayerNorm1(h + h1)
        return h


class Encoder2(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.dropout = 0.5
        self.device = torch.device(args.device)
        self.hidden_size = args.hidden_size
        self.emb_size = args.emb_size
        self.gcn1 = GCNConv(self.hidden_size, self.hidden_size)
        self.LayerNorm1 = nn.LayerNorm([self.hidden_size], elementwise_affine=False)

    def forward(self, x, edge_index, batch):
        x1 = self.gcn1(x, edge_index)
        x = self.LayerNorm1(x1 + x)
        return x


class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean', device='cpu'):
        """
        :param alpha: 权重系数列表
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.device = torch.device(device)
        self.alpha = torch.tensor(alpha).to(self.device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1)  # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  # 对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
