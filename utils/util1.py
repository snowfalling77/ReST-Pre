import json
import warnings

import dgl
import networkx
import networkx as nx
import numpy as np
import torch
from pynvml import *
from scipy import sparse as sp
from scipy.sparse.linalg import ArpackError


def laplacian_positional_encoding(g, pos_enc_dim):  # networkX
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = torch.tensor(g.get_adjacency().data)
    N = sp.diags(dgl.backend.asnumpy(torch.tensor(g.degree())).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.vcount()) - N * A * N

    # Eigenvectors with scipy
    try:
        EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
        EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    except ArpackError as e:
        print("ArpackError occured")
        EigVec = L[:, : pos_enc_dim + 1]
    return torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float()


def get_graph_representation(G, device, hidden_label="H"):
    g_batch = [[[v[hidden_label] for v in g.vs], [[e.source for e in g.es], [e.target for e in g.es]]] for g in G]
    x, edge_index, batch = [], [[], []], []
    pre_len = 0
    for s, g in enumerate(g_batch):
        x.extend(g[0])
        edge_index[0].extend([t + pre_len for t in g[1][0]])
        edge_index[1].extend([t + pre_len for t in g[1][1]])
        batch.extend([s] * len(g[0]))
        pre_len += len(g[0])
    if not x:
        return [], None, None
    x = torch.cat(x, dim=0)
    edge_index = torch.as_tensor(edge_index, device=device, dtype=torch.long)
    batch = torch.as_tensor(batch, device=device)
    return x, edge_index, batch


def event_match(g, S_event):
    # 查看S中的每个节点在g中是否存在，是一个百分比
    S_event.vs["type"] = S_event.vs["type"]
    g.vs["type"] = g.vs["type"]
    count = 0
    for i in range(S_event.vcount()):
        res = g.vs.select(type=S_event.vs[i]["type"])
        if res:
            count = count + 1
    match = count / S_event.vcount()
    return match


def event_sequence_match(g, S_e, l):
    # 查看S中的每个长度为l的序列在g中是否存在，是一个百分比
    S_e.vs["type"] = S_e.vs["type"]
    # 得到S中所有的序列，查看在g中是否存在
    s_sequence_list = [[S_e.es[i].source, S_e.es[i].target]
                       for i in S_e.es.indices]
    g_sequence_list = [[g.es[i].source, g.es[i].target] for i in g.es.indices]
    if l == 2:
        s_list = [[S_e.vs[s]["type"], S_e.vs[t]["type"]] for s, t in s_sequence_list]
        g_list = [[g.vs[s]["type"], g.vs[t]["type"]] for s, t in g_sequence_list]
        count = 0
        for e in s_list:
            if e in g_list:
                count += 1
        match = count / len(s_list)
        return match
    elif l == 3:
        # [s1, t1], [s2, t2] 找 (s1 == t2) or (s2 == t1)
        s_list = [[S_e.vs[s2]["type"], S_e.vs[t2]["type"], S_e.vs[t1]["type"]]
                  for s1, t1 in s_sequence_list for s2, t2 in s_sequence_list if s1 == t2]
        g_list = [[g.vs[s2]["type"], g.vs[t2]["type"], g.vs[t1]["type"]]
                  for s1, t1 in g_sequence_list for s2, t2 in g_sequence_list if s1 == t2]
        count = 0
        for e in s_list:
            if e in g_list:
                count += 1
        match = count / len(s_list)
        return match
    return 0


def MCS(g, s_e):
    """
    两个图的一个最大公共子图是两个图的一个导出子图，它有尽可能多的节点。
    最大公共子图的大小可以反映两个图之间的全局结构相似性。
    因此，我们计算了模式与每个测试实例图之间的最大公共子图的节点数和边数
    :param g:
    :param s:
    :return:
    """
    # 转换为 networkx
    s_types = list(set([v["type"] for v in s_e.vs]))
    g_types = list(set([v["type"] for v in g.vs]))
    types = list(set(s_types + g_types))
    g_edges = []
    for ii, ee in enumerate(g.es):
        edge = [types.index(g.vs[ee.source]["type"]) + 1, types.index(g.vs[ee.target]["type"]) + 1]
        if edge not in g_edges:
            g_edges.append(edge)
    s_edges = []
    for ii, ee in enumerate(s_e.es):
        edge = [types.index(s_e.vs[ee.source]["type"]) + 1, types.index(s_e.vs[ee.target]["type"]) + 1]
        if edge not in s_edges:
            s_edges.append(edge)
    g1 = nx.DiGraph()
    g1.add_edges_from(g_edges)
    s1 = nx.DiGraph()
    s1.add_edges_from(s_edges)
    mcs = getMCS(g1, s1)
    return mcs.number_of_nodes(), mcs.number_of_edges()


def getMCS(g1, g2):
    matching_graph = networkx.Graph()

    for n1, n2 in g2.edges():
        if g1.has_edge(n1, n2):
            matching_graph.add_edge(n1, n2)

    components = networkx.connected_components(matching_graph)

    try:
        largest_component = max(components, key=len)
    except Exception as e:
        # print(e)
        return nx.DiGraph()
    return networkx.induced_subgraph(matching_graph, largest_component)


def show_gpu(simlpe=True):
    # 初始化
    nvmlInit()
    # 获取GPU个数
    deviceCount = nvmlDeviceGetCount()
    total_memory = 0
    total_free = 0
    total_used = 0
    gpu_name = ""
    gpu_num = deviceCount

    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        gpu_name = nvmlDeviceGetName(handle)
        # 查看型号、显存、温度、电源
        if not simlpe:
            print("[ GPU{}: {}".format(i, gpu_name), end="    ")
            print("总共显存: {}G".format((info.total // 1048576) / 1024), end="    ")
            print("空余显存: {}G".format((info.free // 1048576) / 1024), end="    ")
            print("已用显存: {}G".format((info.used // 1048576) / 1024), end="    ")
            print("显存占用率: {}%".format(info.used / info.total), end="    ")
            print("运行温度: {}摄氏度 ]".format(nvmlDeviceGetTemperature(handle, 0)))

        total_memory += (info.total // 1048576) / 1024
        total_free += (info.free // 1048576) / 1024
        total_used += (info.used // 1048576) / 1024

    print("显卡名称：[{}]，显卡数量：[{}]，总共显存；[{}G]，空余显存：[{}G]，"
          "已用显存：[{}G]，显存占用率：[{}%]。".format(
        gpu_name, gpu_num, total_memory, total_free, total_used,
        (total_used / total_memory)
    ))

    # 关闭管理工具
    nvmlShutdown()


def binary_conversion(var: int):
    """
    二进制单位转换
    :param var: 需要计算的变量，bytes值
    :return: 单位转换后的变量，kb 或 mb
    """
    assert isinstance(var, int)
    if var <= 1024 ** 2:
        return f'占用 {round(var / 1024, 2)} KB内存'
    else:
        return f'占用 {round(var / (1024 ** 2), 2)} MB内存'
