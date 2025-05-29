
import re

import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from transformers import XLNetTokenizer, XLNetModel
import pickle
import numpy as np

tokenizer = XLNetTokenizer.from_pretrained(pretrained_model_name_or_path="./xlnet/")
model = XLNetModel.from_pretrained(pretrained_model_name_or_path='./xlnet/')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

file_path = "./data/origin/kairos-ontology.xlsx"

events_ontology = pd.read_excel(file_path, sheet_name="events")

defs = []
type2id = {}
ontology = {
    "events": {},
}
arguments = []

e_row = events_ontology.shape[0]
for i in range(e_row):
    e_type = events_ontology.loc[i, "Type"] + "." + \
             events_ontology.loc[i, "Subtype"] + "." + \
             events_ontology.loc[i, "Sub-subtype"]
    type2id[e_type] = i
    ontology["events"][e_type] = []
    e_def = events_ontology.loc[i, "Definition"]
    defs.append(e_def)
    template = events_ontology.loc[i, "Template"]
    args = re.findall("<(.*?)>", template)
    for arg in args:
        arg_label = events_ontology.loc[i, arg + " label"]
        arg_type = events_ontology.loc[i, arg + " type constraints"]
        ontology["events"][e_type].append((arg_label, arg_type))
        if arg_label not in arguments:
            arguments.append(arg_label)

type2id["start"] = e_row
defs.append("Start nodes of all events in the event graph")

type2id["end"] = e_row + 1
defs.append("End nodes of all events in the event graph")

type_tensors = tokenizer(defs, padding=True, truncation=True, return_tensors="pt")
type_embeddings = []

for i, j in enumerate(tqdm(defs)):

    with torch.no_grad():
        output = model(input_ids=type_tensors["input_ids"][i:i + 1].to(device),
                       attention_mask=type_tensors['attention_mask'][i:i + 1].to(device))
        # pooler_output = output['pooler_output']
        last_hidden_state = output["last_hidden_state"]
        output1 = last_hidden_state[:, -1]
    type_embeddings.append(output1)

type_embeddings = torch.cat(type_embeddings, dim=0)

print(type_embeddings.shape)

with open("./data/prepared/ontology_embeddings.pkl", "wb") as f:
    pickle.dump((ontology, type2id, type_embeddings), f)

