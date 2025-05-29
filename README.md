# ReST-Pre
ReST-Pre: Event Prediction by spatial-temporal Structural Replay on Generative Implicit Event Pattern Induction

The source data comes from https://github.com/limanling/temporal-graph-schema, and the corresponding data is placed in the `data/origin directory`

The file structure is as follows:

```
G:.
│  .gitattributes
│  .gitignore
│  data_prepare.py
│  embedding.py
│  README.md
│  run.py
│
├─data
│  ├─origin
│  │  │  kairos-ontology.xlsx
│  │  │
│  │  ├─LDC_schema_corpus_ce_split
│  │  │  ├─dev
│  │  │  │      001.dev.json
│  │  │  │      002.dev.json
│  │  │  │      003.dev.json
│  │  │  │      ...
│  │  │  │
│  │  │  ├─test
│  │  │  │      001.test.json
│  │  │  │      002.test.json
│  │  │  │      003.test.json
│  │  │  │      ...
│  │  │  │      
│  │  │  └─train
│  │  │          001.train.json
│  │  │          002.train.json
│  │  │          003.train.json
│  │  │          ...
│  │  │
│  │  ├─RESIN_schema
│  │  │      resin-schemalib-readme.md
│  │  │      resin-schemalib.json
│  │  │
│  │  └─Wiki_IED_split
│  │      ├─dev
│  │      │      suicide_ied_dev.json
│  │      │      wiki_drone_strikes_dev.json
│  │      │      wiki_ied_bombings_dev.json
│  │      │      wiki_mass_car_bombings_dev.json
│  │      │
│  │      ├─test
│  │      │      suicide_ied_test.json
│  │      │      wiki_drone_strikes_test.json
│  │      │      wiki_ied_bombings_test.json
│  │      │      wiki_mass_car_bombings_test.json
│  │      │
│  │      └─train
│  │              suicide_ied_train.json
│  │              wiki_drone_strikes_train.json
│  │              wiki_ied_bombings_train.json
│  │              wiki_mass_car_bombings_train.json
│  │
│  ├─prepared
│  │  │  ontology_embeddings.pkl
│  │  │
│  │  ├─IED
│  │  │  ├─dev
│  │  │  │      node12.json
│  │  │  │      node16.json
│  │  │  │      node8.json
│  │  │  │
│  │  │  ├─test
│  │  │  │      node12.json
│  │  │  │      node16.json
│  │  │  │      node8.json
│  │  │  │
│  │  │  └─train
│  │  │          node12.json
│  │  │          node16.json
│  │  │          node8.json
│  │  │
│  │  └─LDC
│  │      ├─dev
│  │      │      node12.json
│  │      │      node16.json
│  │      │      node8.json
│  │      │
│  │      ├─test
│  │      │      node12.json
│  │      │      node16.json
│  │      │      node8.json
│  │      │
│  │      └─train
│  │              node12.json
│  │              node16.json
│  │              node8.json
│  │
├─models
│  │  graph_generation.py
│  │  graph_replay.py
│  │  ours.py
│
├─tf-logs
├─utils
│  │  func.py
│  │  ontology.py
│  │  util.py
│  │  util1.py
│
└─xlnet
        config.json
        pytorch_model.bin
        special_tokens_map.json
        spiece.model
        tokenizer_config.json
```
