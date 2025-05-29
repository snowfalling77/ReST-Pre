import json
import math
import os.path

import igraph
import torch
from torch import nn


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def getBCELoss(input, target, device):
    count = 1
    for sh in target.shape:
        count *= sh
    weight_ratio = torch.zeros(2, device=device, dtype=torch.float32)
    weight_ratio[0] = torch.sum(target)
    weight_ratio[1] = count - torch.sum(target)
    # weight_ratio = torch.softmax(weight_ratio, dim=0)
    weight_ratio = weight_ratio / target.shape[0]
    weight = torch.zeros_like(target).float().to(device)
    weight = torch.fill_(weight, weight_ratio[0])
    weight[target > 0] = weight_ratio[1]
    return nn.BCELoss(weight=weight, reduction="sum")(input, target)


def getCELoss(input, target, device):
    nn.CrossEntropyLoss()
    weight_ratio = torch.zeros(2, device=device, dtype=torch.float32)
    weight_ratio[0] = torch.sum(target)
    weight_ratio[1] = target.shape[0] - torch.sum(target)
    weight_ratio = torch.softmax(weight_ratio, dim=0)
    weight = torch.zeros_like(target).float().to(device)
    weight = torch.fill_(weight, weight_ratio[0])
    weight[target > 0] = weight_ratio[1]
    return nn.CrossEntropyLoss(weight=weight)(input, target)


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


def print2file(file_name, obj, type_="obj"):
    file_dirs = file_name.split("/")
    for i in range(1, len(file_dirs)):
        file_dir = "/".join(file_dirs[: i])
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
    with open(file_name, "a+", encoding="utf-8") as f:
        if type_ == "obj":
            f.write(str(obj))
        elif type_ == "model":
            for param_tensor in obj.state_dict():
                f.write(str(param_tensor))
                f.write("\n")
                f.write(str(obj.state_dict()[param_tensor]))
                f.write("\n")
        f.write("\n\n==========\n\n")


def load_schema(path):
    S = {}
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
        data = json.loads(content)
        schemas = data["schemas"]
        for sc in schemas:
            name = sc["name"].lower()
            steps = sc["steps"]
            order = sc["order"]
            entityRelations = sc["entityRelations"]
            g = igraph.Graph(directed=True, n=len(steps))
            if g.vcount() == 0:
                return None
            g["name"] = name

            for i in range(len(steps)):
                g.vs[i]["isEvent"] = True
                g.vs[i]["iid"] = steps[i]["@id"]
                g.vs[i]["@type"] = steps[i]["@type"].split("/")[-1]
                g.vs[i]["trigger"] = steps[i]["name"]
                participants = steps[i]["participants"]
                for participant in participants:
                    g.add_vertex(role=participant["role"].split("/")[-1], iid=participant["@id"], isEvent=False)
                    g.add_edge(
                        i, g.vcount() - 1,
                        role=participant["role"].split("/")[-1],
                        property="argument"
                    )

            for i in range(len(order)):
                e_b_id = [
                    j for j in range(g.vcount())
                    if g.vs[j]["isEvent"] and g.vs[j]["iid"] in order[i]["before"]
                ]
                e_a_id = [
                    j for j in range(g.vcount())
                    if g.vs[j]["isEvent"] and g.vs[j]["iid"] in order[i]["after"]
                ]
                for ii in e_b_id:
                    for jj in e_a_id:
                        g.add_edge(ii, jj, property="temporal")

            for i in range(len(entityRelations)):
                v_s_id = [
                    j for j in range(g.vcount())
                    if not g.vs[j]["isEvent"] and g.vs[j]["iid"] == entityRelations[i]["relationSubject"]
                ][0]  # Subject
                v_o_ids = [
                    (j, r["relationPredicate"].split("/")[-1]) for j in range(g.vcount())
                    for r in entityRelations[i]["relations"] for r1 in r["relationObject"]
                    if not g.vs[j]["isEvent"] and g.vs[j]["iid"] == r1
                ]  # Object
                for v_o_id, type1 in v_o_ids:
                    g.add_edge(
                        v_s_id, v_o_id,
                        type=type1,
                        property="entityRelation"
                    )
            S[name] = g
        return S


def check_nan(tensor):
    has_nan = torch.any(torch.isnan(tensor))
    print("是否存在 NaN:", has_nan.item())  # 输出 True
    return has_nan
