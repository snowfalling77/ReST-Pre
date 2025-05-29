import pickle

from torch import nn

from models.graph_generation import GraphGenerator
from models.graph_replay import GraphReplay


class ReST(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.graph_generator = GraphGenerator(args)
        with open(args.ontology_path, "rb") as f:
            ontology_embeddings = pickle.load(f)
        self.ontology, self.type2id, self.type_embeddings = ontology_embeddings
        self.graph_replay = GraphReplay(args)

    def forward(self, g_batch):
        G_re, loss1 = self.graph_generator(g_batch)
        for i in range(len(g_batch)):
            G_re[i]["target"] = g_batch[i]["target"]
            G_re[i]["target_index"] = g_batch[i]["target_index"]
            G_re[i]["candidate"] = g_batch[i]["candidate"]
            for j in range(G_re[i].vcount()):
                G_re[i].vs[j]["H"] = self.type_embeddings[self.type2id[G_re[i].vs[j]["type"]]].view(1, -1)
        loss2, count = self.graph_replay(G_re)
        all_loss = loss1 + loss2
        return G_re, all_loss, count

    def forward_test(self, g_batch):
        G_re, loss1 = self.graph_generator.forward_test(g_batch)
        for i in range(len(g_batch)):
            G_re[i]["target"] = g_batch[i]["target"]
            G_re[i]["target_index"] = g_batch[i]["target_index"]
            G_re[i]["candidate"] = g_batch[i]["candidate"]
            for j in range(G_re[i].vcount()):
                G_re[i].vs[j]["H"] = self.type_embeddings[self.type2id[G_re[i].vs[j]["type"]]].view(1, -1)
        loss2, count = self.graph_replay(G_re)
        all_loss = loss1 + loss2
        return G_re, all_loss, count
