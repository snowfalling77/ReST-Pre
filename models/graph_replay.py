import pickle

import torch
import torch.nn.functional as F
from torch import nn

from utils.func import check_nan


class GraphReplay(nn.Module):

    def __init__(self, args):
        super().__init__()
        new_dilation = 1
        self.args = args
        self.device = torch.device(args.device)
        self.sch_init = nn.Sequential(
            nn.Linear(args.emb_size, args.hidden_size),
            nn.Dropout(0.5),
            nn.LayerNorm([args.hidden_size], elementwise_affine=False)
        )
        self.filter_convs = dilated_inception(args.residual_channels, args.dilation_channels, dilation_factor=new_dilation)
        self.gate_convs = dilated_inception(args.residual_channels, args.dilation_channels, dilation_factor=new_dilation)
        self.start_conv = nn.Conv2d(in_channels=1, out_channels=args.residual_channels, kernel_size=(1, 1))
        self.end_conv_x = nn.Conv2d(in_channels=args.skip_channels, out_channels=args.end_channels, kernel_size=(1, 1), bias=True)
        self.layer_norm1 = nn.LayerNorm([args.hidden_size - 6], elementwise_affine=False)
        self.layer_norm2 = nn.LayerNorm([args.hidden_size - 6], elementwise_affine=False)
        with open(args.ontology_path, "rb") as f:
            ontology_embeddings = pickle.load(f)
        self.ontology, self.type2id, self.type_embeddings = ontology_embeddings
        self.graph_propagation = GraphPropagation(
            self.ontology, self.type2id, self.type_embeddings, args.hidden_size, args.emb_size, args.device
        )
        self.gat_layer = GATLayer(args.hidden_size, 8 * (args.hidden_size - 6))
        self.spatio_end_layer = nn.Linear(in_features=(args.hidden_size - 6) * 8, out_features=args.hidden_size - 6)
        self.embedding_layer = nn.Sequential(nn.Linear(in_features=(args.hidden_size - 6) * 2 * (args.graph_size + 2), out_features=args.emb_size))
        self.pred_layer = nn.Sequential(
            nn.Linear(in_features=(args.hidden_size - 6) * 2 + args.emb_size * args.candidate_size, out_features=args.candidate_size),
            nn.Softmax(dim=-1)
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, g_batch):
        self.init_graph(g_batch)
        # 时空重放
        x1 = self.layer_norm1(self.temporal_replay(g_batch))
        # check_nan(x1)
        x2 = self.layer_norm2(self.spatio_replay(g_batch))
        # check_nan(x2)
        x = torch.cat([x1, x2], dim=-1)
        x = torch.mean(x, dim=1)
        # pred_embs = self.embedding_layer(x.view(len(g_batch), -1))
        candidate_embs = torch.cat([torch.cat(
            [self.type_embeddings[self.type2id[ct]].view(1, -1) for ct in g["candidate"]], dim=-1
        ) for g in g_batch], dim=0)
        pred_res = self.pred_layer(torch.cat([x.view(len(g_batch), -1), candidate_embs], dim=-1))
        target_ids = [g["target_index"] for g in g_batch]
        true_ids = torch.tensor(target_ids, dtype=torch.int64, device=self.device)
        # target_embs = [self.type_embeddings[id].view(1, -1) for id in target_ids]
        # target_embs = torch.cat(target_embs, dim=0)
        # loss, count = compute_metrics(pred_embs, self.type_embeddings, true_ids)
        loss, count = self.calculate_loss_and_accuracy(pred_res, true_ids)
        return loss, count

    def calculate_loss_and_accuracy(self, predictions, targets):
        """
        参数:
        predictions : torch.Tensor 形状为 (batch_size, 5) 的预测logits
        targets     : torch.Tensor 形状为 (batch_size,) 的真实类别索引

        返回:
        loss        : torch.Tensor 标量损失值
        correct_num : int 预测正确的样本数量
        """
        # 计算交叉熵损失（自带softmax处理）
        loss = self.loss_fn(predictions, targets)

        # 获取预测类别（取概率最大的索引）
        _, predicted_labels = torch.max(predictions, dim=1)

        # 统计正确预测数量
        correct_num = (predicted_labels == targets).sum().item()

        return loss, correct_num

    def temporal_replay(self, g_batch):
        # DIC
        # (batch_size, node_count, hidden_size)
        X = []
        for g in g_batch:
            x = torch.cat([v["H"] for v in g.vs], dim=0)
            X.append(x.unsqueeze(0))
        x1 = torch.cat(X, dim=0)
        x1 = x1.unsqueeze(1)
        X = self.start_conv(x1)
        filter = self.filter_convs(X)
        filter = torch.tanh(filter)
        gate = self.gate_convs(X)
        gate = torch.sigmoid(gate)
        x = filter * gate
        x = self.end_conv_x(x).squeeze(1)
        return x

    def spatio_replay(self, g_batch):
        g_batch = self.graph_propagation(g_batch)
        feature_tensor, adj_tensor = self.get_batch_tensors(g_batch)
        x = self.gat_layer(feature_tensor, adj_tensor)
        x = self.spatio_end_layer(x)
        return x

    def init_graph(self, g_batch):
        for i, g in enumerate(g_batch):
            for v in range(g.vcount()):
                H = self.sch_init(g.vs[v]["H"].to(self.device))
                g_batch[i].vs[v]["H"] = H

    def find_closest_type_ids(self, input_vecs):
        """
        找到输入向量在type_embeddings中余弦相似度最高的类型ID
        输入：
            input_vecs (16, 768)
            type_embeddings (69, 768)
        输出：包含16个ID的列表
        """
        # 扩展维度用于广播计算
        input_expanded = input_vecs.unsqueeze(1)  # (16, 1, 768)
        type_expanded = self.type_embeddings.unsqueeze(0)  # (1, 69, 768)

        # 计算所有组合的余弦相似度
        similarities = F.cosine_similarity(input_expanded, type_expanded, dim=2)  # (16, 69)

        # 获取每个输入向量对应的最大相似度索引
        closest_ids = torch.argmax(similarities, dim=1)  # (16,)

        return closest_ids.tolist()

    def get_batch_tensors(self, g_batch):
        """
        返回形状为(batch_size, max_N, in_dim)的特征张量
        和(batch_size, max_N, max_N)的邻接矩阵张量
        """
        # 确定批次中的最大节点数和特征维度
        max_N = max(g.vcount() for g in g_batch) if g_batch else 0
        in_dim = self.args.hidden_size
        # 初始化批量张量
        batch_size = len(g_batch)
        feature_tensor = torch.zeros((batch_size, max_N, in_dim),
                                     dtype=torch.float32, device=self.device)
        adj_tensor = torch.zeros((batch_size, max_N, max_N),
                                 dtype=torch.float32, device=self.device)

        for i, g in enumerate(g_batch):
            n = g.vcount()
            if n == 0:
                continue  # 保持全零填充

            # 处理节点特征
            node_features = torch.cat(g.vs["H"], dim=0)
            feature_tensor[i, :n] = node_features

            # 处理邻接矩阵
            adj = torch.tensor(g.get_adjacency().data,
                               dtype=torch.float32,
                               device=self.device)
            adj_tensor[i, :n, :n] = adj

        return feature_tensor, adj_tensor


class GraphPropagation(nn.Module):
    ablation = 3
    device = torch.device("cpu")
    max_len = 90

    def __init__(self, ontology, type2id, type_embeddings, hidden_size, emb_size, device):
        super().__init__()
        self.latent_size = 64
        self.dropout = 0.5
        self.rate = 0.5 - 0.5
        self.device = device
        self.type2id, self.type_embeddings = type2id, type_embeddings
        self.type_list = [kk for kk in ontology["events"]]
        self.e_embedding = torch.cat([self.type_embeddings[self.type2id[kk]].view(1, -1) for kk in self.type_list],
                                     dim=0)
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.LeakyReLU = nn.LeakyReLU()
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()
        self.layer_norm = nn.LayerNorm([self.hidden_size], elementwise_affine=False)
        self.attention_forward = nn.MultiheadAttention(self.hidden_size, 1, dropout=self.dropout, batch_first=True)
        self.attention_backward = nn.MultiheadAttention(self.hidden_size, 1, dropout=self.dropout, batch_first=True)
        self.loss_function = nn.MSELoss(reduction="sum")

    def create_matrix(self, dims: tuple):
        return nn.Parameter(torch.randn(dims, device=self.device))

    def forward(self, G):
        self.propagate_backward(G, max([g.vcount() for g in G]))
        return G

    def propagate_forward(self, G, max_len, start=0):
        for v in range(start, max_len):
            self._propagate_forward(G, v)

    def propagate_backward(self, G, max_len, start=0):
        for v in range(start, max_len):
            self._propagate_backward(G, v)

    def _propagate_forward(self, G, v):
        G = [g for g in G if g.vcount() > v]
        if not G:
            return
        H_pre = [[g.vs[p]["H"] for p in g.predecessors(v)] for g in G]
        max_len = max([len(p) for p in H_pre])
        H_v = torch.cat([g.vs[v]["H"].unsqueeze(0) for g in G], dim=0)
        if max_len != 0:
            H_pre = torch.cat([torch.cat(h_pre + [
                torch.zeros((max_len - len(h_pre), self.hidden_size), device=self.device)
            ], dim=0).unsqueeze(0) for h_pre in H_pre], dim=0)
            attn_output, attn_weight = self.attention_forward(H_v, H_pre, H_pre)
            # H_v = self.MLP_forward(torch.cat([H_v, attn_output], dim=-1))
            H_v = self.layer_norm(attn_output + H_v)
        for i, g in enumerate(G):
            g.vs[v]["H"] = H_v[i]
        return

    def _propagate_backward(self, G, v):
        G = [g for g in G if g.vcount() > v]
        if not G:
            return
        H_next = [[g.vs[p]["H"] for p in g.successors(v)] for g in G]
        max_len = max([len(p) for p in H_next])
        H_v = torch.cat([g.vs[v]["H"].unsqueeze(0) for g in G], dim=0)
        if max_len != 0:
            H_next = torch.cat([torch.cat(h_next + [
                torch.zeros((max_len - len(h_next), self.hidden_size), device=self.device)
            ], dim=0).unsqueeze(0) for h_next in H_next], dim=0)
            attn_output, attn_weight = self.attention_backward(H_v, H_next, H_next)
            # H_v = self.MLP_backward(torch.cat([H_v, attn_output], dim=-1))
            H_v = self.layer_norm(attn_output + H_v)
        for i, g in enumerate(G):
            g.vs[v]["H"] = H_v[i]
        return

    def getBCELoss(self, input, target):
        weight_ratio = torch.zeros(2, device=self.device, dtype=torch.float32)
        weight_ratio[0] = torch.sum(target)
        weight_ratio[1] = target.shape[0] - torch.sum(target)
        # weight_ratio = torch.softmax(weight_ratio, dim=0)
        weight_ratio = weight_ratio / target.shape[0]
        weight = torch.zeros_like(target).float().to(self.device)
        weight = torch.fill_(weight, weight_ratio[0])
        weight[target > 0] = weight_ratio[1]
        return nn.BCELoss(weight=weight, reduction="sum")(input, target)


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, adjacency_matrix, node_features):
        # 使用邻接矩阵进行传播
        x = torch.matmul(adjacency_matrix, node_features)
        x = self.linear(x)
        x = F.relu(x)  # 激活函数可以根据任务调整

        return x


class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=2):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=8, concat=True):
        super(GATLayer, self).__init__()
        self.heads = heads
        self.concat = concat
        self.out_dim = out_dim // heads  # 每个 head 的输出维度
        self.W = nn.Linear(in_dim, out_dim, bias=False)  # 线性变换
        self.a = nn.Linear(2 * self.out_dim, 1, bias=False)  # 修改为每个头产生1个分数
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, adj, edge_weight=None):
        """
        x: (batch, N, in_dim) -> 节点特征
        adj: (batch, N, N) -> 邻接矩阵
        edge_weight: (batch, N, N) -> 边的权重 (可选)
        """
        B, N, _ = x.shape
        identity = torch.eye(N, device=adj.device).unsqueeze(0).expand(B, -1, -1)
        adj = torch.clamp(adj + identity, 0, 1)  # 确保自环存在

        # 线性变换并分割头
        x_transformed = self.W(x).view(B, N, self.heads, self.out_dim)  # (B, N, H, F)

        # 生成所有节点对的特征组合
        x_i = x_transformed.unsqueeze(2)  # (B, N, 1, H, F)
        x_j = x_transformed.unsqueeze(1)  # (B, 1, N, H, F)

        # 拼接特征并计算注意力分数
        x_pairs = torch.cat([x_i.expand(-1, -1, N, -1, -1),
                             x_j.expand(-1, N, -1, -1, -1)], dim=-1)  # (B, N, N, H, 2F)
        e = self.leaky_relu(self.a(x_pairs)).squeeze(-1)  # (B, N, N, H)

        # 应用边权重
        if edge_weight is not None:
            e = e * edge_weight.unsqueeze(-1)  # (B, N, N, H)

        # 创建邻接mask并应用
        adj_mask = adj.unsqueeze(-1).expand(-1, -1, -1, self.heads)  # (B, N, N, H)
        e = e.masked_fill(adj_mask == 0, float('-inf'))

        # 添加数值稳定性处理
        max_values = e.max(dim=2, keepdim=True)[0].detach()
        e_stable = torch.where(adj_mask == 0, torch.zeros_like(e), e - max_values)
        alpha = F.softmax(e_stable, dim=2) # (B, N, N, H)

        # 特征聚合（关键修正部分）
        alpha = alpha.permute(0, 3, 1, 2)  # (B, H, N, N)
        x_transformed = x_transformed.permute(0, 2, 1, 3)  # (B, H, N, F)

        # 使用注意力权重进行加权求和
        h = torch.matmul(alpha, x_transformed)  # (B, H, N, F)
        h = h.permute(0, 2, 1, 3)  # (B, N, H, F)

        # 处理concat选项
        if self.concat:
            h = h.reshape(B, N, -1)  # 合并头
        else:
            h = h.mean(dim=2)  # 平均头

        return F.relu(h)


class SimilarityClassifier(nn.Module):
    def __init__(self, class_vectors):
        """
        Args:
            class_vectors: 预定义的类别向量 [n_classes, H]
        """
        super().__init__()
        # 注册不参与梯度计算的缓冲区
        self.register_buffer('class_vectors', class_vectors)  # [n_classes, H]
        self.n_classes = class_vectors.size(0)

    def forward(self, pred_vectors):
        """
        Args:
            pred_vectors: 预测向量 [batch_size, H]
        Returns:
            similarities: 相似度矩阵 [batch_size, n_classes]
        """
        # 余弦相似度计算（自动广播机制）
        pred_norm = F.normalize(pred_vectors, p=2, dim=1)  # [batch, H]
        class_norm = F.normalize(self.class_vectors, p=2, dim=1)  # [n_classes, H]
        similarities = torch.mm(pred_norm, class_norm.T)  # [batch, n_classes]
        return similarities


def compute_metrics(pred_vectors, class_vectors, true_ids):
    """
    计算损失和正确数量

    Args:
        pred_vectors: 预测向量 [batch, H]
        class_vectors: 类别向量 [n_classes, H]
        true_ids: 真实类别ID [batch]

    Returns:
        loss: 标量损失值
        correct: 正确预测数量
    """
    # 初始化相似度计算模块
    sim_calculator = SimilarityClassifier(class_vectors)

    # 计算相似度矩阵
    logits = sim_calculator(pred_vectors)  # [batch, n_classes]

    # 计算交叉熵损失
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, true_ids)

    # 计算正确数量
    pred_ids = torch.argmax(logits, dim=1)  # [batch]
    correct = torch.sum(pred_ids == true_ids).item()

    return loss, correct


