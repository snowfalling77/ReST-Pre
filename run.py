import argparse
import glob
import json
import os
import pickle
import random
import time

import igraph
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import optim

from models.ours import ReST
from utils.util1 import event_match, event_sequence_match, MCS, show_gpu

parser = argparse.ArgumentParser(description='Train Model')

parser.add_argument('--is-log', action='store_true', default=True, help='if True, save log')
parser.add_argument('--is-gpu', action='store_true', default=True, help='')
parser.add_argument('--is-train', action='store_true', default=True, help='')
parser.add_argument('--dataset-name', type=str, default="IED", metavar='LR', help='dataset name')
parser.add_argument('--log-dir', type=str, default="./tf-logs", metavar='LR', help='log name')
parser.add_argument('--log-name', type=str, default="", metavar='LR', help='log name')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR', help='learning rate')
parser.add_argument('--pos_enc_dim', type=int, default=6, metavar='N', help='')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--graph_size', type=int, default=8, metavar='N', help='')
parser.add_argument('--candidate_size', type=int, default=12, metavar='N', help='')
parser.add_argument('--encoder_count', type=int, default=1, metavar='N', help='')
parser.add_argument('--decoder_count', type=int, default=1, metavar='N', help='')
parser.add_argument('--L', action='store_true', default=False, help='')
parser.add_argument('--R', action='store_true', default=False, help='')
parser.add_argument('--batch-size', type=int, default=16, metavar='N', help='batch size during training')
parser.add_argument('--hidden-size', type=int, default=128, metavar='N', help='hidden size during training')
parser.add_argument('--emb-size', type=int, default=768, metavar='N', help='hidden size during training')
parser.add_argument('--residual-channels', type=int, default=16, metavar='N', help='hidden size during training')
parser.add_argument('--skip-channels', type=int, default=16, metavar='N', help='hidden size during training')
parser.add_argument('--end-channels', type=int, default=1, metavar='N', help='hidden size during training')
parser.add_argument('--dilation-channels', type=int, default=16, metavar='N', help='hidden size during training')
parser.add_argument('--seed', type=int, default=43, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

args.ontology_path = "./data/prepared/ontology_embeddings.pkl"
args.dataset_dir = "./data/prepared/" + args.dataset_name


args.device = 'cpu'
if args.is_gpu and torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:0'

print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

log_name = ""

writer = None
if args.is_log:
    writer = SummaryWriter(log_dir=args.log_dir)
    if args.log_name == "":
        log_name = str(time.strftime("%Y-%m-%d %H-%M-%S")) + args.dataset_name
    else:
        temp = args.log_name.replace(" ", "_")
        file_list = os.listdir(args.log_dir)
        num = 0
        flag = False
        while not flag:
            if temp + str(num) not in file_list:
                flag = True
            else:
                num += 1
        log_name = temp + str(num)


def train(epoch):
    print("=" * 30, "train epoch:", epoch, "=" * 30)
    g_batch = []
    all_loss = 0
    a, b = 0, 0
    for i, g in enumerate(train_data):
        g_batch.append(g)
        if len(g_batch) == args.batch_size or i == len(train_data) - 1:
            G_re, loss, count = model(g_batch)
            a += count
            b += len(g_batch)
            all_loss += loss
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            g_batch = []
    if args.is_log:
        writer.add_scalars("{}/train/loss".format(log_name), {"loss": all_loss / len(train_data)}, epoch)
    print(a / b)
    return all_loss / len(train_data)


def test(epoch):
    print("=" * 30, "test epoch:", epoch, "=" * 30)
    g_batch = []
    all_loss = 0
    model.training = False
    event_match_p, event_sequence_match2, event_sequence_match3 = 0, 0, 0
    mcs_nodes, mcs_edges = 0, 0
    for i, g in enumerate(test_data):
        g_batch.append(g)
        if len(g_batch) == args.batch_size or i == len(test_data) - 1:
            G_re, loss, count = model.forward_test(g_batch)
            all_loss += loss
            for g_re, g in zip(G_re, g_batch):
                event_match_p += event_match(g_re, g)
                event_sequence_match2 += event_sequence_match(g_re, g, 2)
                event_sequence_match3 += event_sequence_match(g_re, g, 3)
                mcs1, mcs2 = MCS(g_re, g)
                mcs_nodes += mcs1
                mcs_edges += mcs2
            print(loss)
            g_batch = []
    event_match_p = event_match_p / len(test_data)
    event_sequence_match2 = event_sequence_match2 / len(test_data)
    event_sequence_match3 = event_sequence_match3 / len(test_data)
    mcs_nodes = mcs_nodes / len(test_data)
    mcs_edges = mcs_edges / len(test_data)
    if args.is_log:
        writer.add_scalars("{}/test".format(log_name), {
            "loss": all_loss / len(test_data),
            "event_match_p": event_match_p,
            "event_sequence_match2": event_sequence_match2,
            "event_sequence_match3": event_sequence_match3,
            "mcs_nodes": mcs_nodes,
            "mcs_edges": mcs_edges,
        }, epoch)
    print(
        "====" * 10, "\n",
        "test:",
        "\nevent_match_p:", event_match_p,
        "\nevent_sequence_match2:", event_sequence_match2,
        "\nevent_sequence_match3:", event_sequence_match3,
        "\nmcs_nodes:", mcs_nodes,
        "\nmcs_edges:", mcs_edges, "\n",
        "====" * 10,
    )
    return all_loss / len(test_data)


def dev(epoch):
    print("=" * 30, "dev epoch:", epoch, "=" * 30)
    g_batch = []
    all_loss = 0
    model.training = False
    event_match_p, event_sequence_match2, event_sequence_match3 = 0, 0, 0
    mcs_nodes, mcs_edges = 0, 0
    for i, g in enumerate(dev_data):
        g_batch.append(g)
        if len(g_batch) == args.batch_size or i == len(dev_data) - 1:
            G_re, loss, count = model.forward_test(g_batch)
            all_loss += loss
            for g_re, g in zip(G_re, g_batch):
                event_match_p += event_match(g_re, g)
                event_sequence_match2 += event_sequence_match(g_re, g, 2)
                event_sequence_match3 += event_sequence_match(g_re, g, 3)
                mcs1, mcs2 = MCS(g_re, g)
                mcs_nodes += mcs1
                mcs_edges += mcs2
            print(loss)
            g_batch = []
    event_match_p = event_match_p / len(dev_data)
    event_sequence_match2 = event_sequence_match2 / len(dev_data)
    event_sequence_match3 = event_sequence_match3 / len(dev_data)
    mcs_nodes = mcs_nodes / len(dev_data)
    mcs_edges = mcs_edges / len(dev_data)
    if args.is_log:
        writer.add_scalars("{}/dev".format(log_name), {
            "loss": all_loss / len(dev_data),
            "event_match_p": event_match_p,
            "event_sequence_match2": event_sequence_match2,
            "event_sequence_match3": event_sequence_match3,
            "mcs_nodes": mcs_nodes,
            "mcs_edges": mcs_edges,
        }, epoch)
    print(
        "====" * 10, "\n",
        "dev:",
        "\nevent_match_p:", event_match_p,
        "\nevent_sequence_match2:", event_sequence_match2,
        "\nevent_sequence_match3:", event_sequence_match3,
        "\nmcs_nodes:", mcs_nodes,
        "\nmcs_edges:", mcs_edges, "\n",
        "====" * 10,
    )
    return all_loss / len(dev_data)


def load_ontology():
    with open(args.ontology_path, "rb") as f:  # 注意模式是 "rb"（二进制读取）
        data = pickle.load(f)
    # 解包元组 (ontology, type2id, type_embeddings)
    ontology, type2id, type_embeddings = data
    return type2id, type_embeddings


def load_data(path, type2id, type_emb):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 添加向量表示
    G = []
    for i, d in enumerate(data):
        g = igraph.Graph(directed=True)
        g.add_vertices(len(d["events"]))
        g.add_edges(d["edges"])
        for i1, e in enumerate(d["events"]):
            g.vs[i1]["type"] = e
            g.vs[i1]["oH"] = type_emb[type2id[e]].view(1, -1)
        g["enhanced_edges"] = d["enhanced_edges"]
        g["target"] = d["target"]
        g["target_index"] = d["target_index"]
        g["candidate"] = d["candidate"]
        g["tH"] = type_emb[type2id[d["target"]]].view(1, -1)
        G.append(g)
    return G


if __name__ == '__main__':
    type2id, type_emb = load_ontology()

    train_data = load_data(f'{args.dataset_dir}/train/node{args.graph_size}.json', type2id, type_emb)
    print(f"训练集共有{len(train_data)}条数据")
    test_data = load_data(f'{args.dataset_dir}/test/node{args.graph_size}.json', type2id, type_emb)
    dev_data = load_data(f'{args.dataset_dir}/dev/node{args.graph_size}.json', type2id, type_emb)

    model = ReST(args)
    model.to(torch.device(args.device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.1)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    for epoch in range(args.epochs):
        loss1 = train(epoch)
        loss2 = dev(epoch)
        loss3 = test(epoch)
        if args.is_log:
            writer.add_scalars("{}/loss".format(log_name), {
                "loss_train": loss1,
                "loss_dev": loss2,
                "loss_test": loss3,
            }, epoch)
        show_gpu(simlpe=False)
        ExpLR.step()
    if not os.path.exists("./checkpoint/{}".format(log_name)):
        os.mkdir("./checkpoint/{}".format(log_name))
    torch.save(model.state_dict(), "./checkpoint/{}/model-last-epoch.pt".format(log_name))

