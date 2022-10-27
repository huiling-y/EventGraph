# EventGraph

This repository provides the source code of our paper "EventGraph: Event Extraction as Semantic Graph Parsing".



## How to run

### :feet: &nbsp; Preprocessing

#### ACE-E: DyGIE++ to EventGraph format

To convert DyGIE++ format (https://github.com/dwadden/dygiepp/tree/master/scripts/data/ace-event) to the format used by EventGraph. DyGIE++ creates a `json` folder with `train.json`, `dev.json`, and `test.json` for extracted event mentions.

```shell
cd preprocess

python3 convert_dygie.py --dygie_data_dir <DATA_DIR>
```

#### ACE-E+: OneIE to EventGraph format

To convert OneIE format (`preprocessing/process_ace.py` from [OneIE v0.4.8](https://blender.cs.illinois.edu/software/oneie/)). OneIE generates a `english.json` file for extracted event mentions.

```shell
cd preprocess

python3 convert_oneie.py --oneie_f <FILE>
```

#### ACE++ and ACE+++

Extract event mentions from the original annotations of ACE05 ([LDC2006T6](https://catalog.ldc.upenn.edu/LDC2006T06)). We use CoreNLP for tokenization ([v2018-02-27](http://nlp.stanford.edu/software/stanford-corenlp-full-2018-02-27.zip)). The script processes the files directly from the folder for English (`ace_2005_td_v7/data/English`).


```shell
cd preprocess

python3 extract_ace_events.py --data_dir <ace_2005_td_v7/data/English> --corenlp_dir <CoreNLP_Dir>
```

### :feet: &nbsp; Convert to graph (mrp) format 

To convert event mentions to graph (mrp) format. By default, processed datasets are located in `dataset` folder as following:

```
dataset
├── labeled_edge_mrp
│   ├── ace_en
│   ├── ace_p_en
│   ├── ace_pp_en
│   └── ace_ppp_en
├── raw
│   ├── ace_en
│   ├── ace_p_en
│   ├── ace_pp_en
│   └── ace_ppp_en
├── splits
└── splits2
```

> `ace_en`, `ace_p_en`, `ace_pp_en`, `ace_ppp_en` corresponding to `ACE-E` `ACE-E+`, `ACE-E++`, `ACE-E+++` respectively

Run the following script to convert one dataset.

```shell
sh convert_to_mrp.sh <DATASET>
```

### :feet: &nbsp; Example

- Raw format

```
{'sent_id': 'bc/CNN_IP_20030329.1600.02/001',
 'text': 'It was in northern Iraq today that an eight artillery round hit the site occupied by Kurdish fighters near Chamchamal',
 'events': [{'event_type': 'Attack',
   'trigger': [['hit'], ['60:63']],
   'arguments': [[['today'], ['24:29'], 'Time-Within'],
    [['northern Iraq'], ['10:23'], 'Place'],
    [['the site occupied by Kurdish fighters near Chamchamal'],['64:117'], 'Target'],
    [['an eight artillery round'], ['35:59'], 'Instrument']]}]}

```

- MRP format

```
{'id': 'bc/CNN_IP_20030329.1600.02/001',
 'flavor': 1,
 'framework': 'ace',
 'version': 1.1,
 'time': '2022-10-19',
 'input': 'It was in northern Iraq today that an eight artillery round hit the site occupied by Kurdish fighters near Chamchamal',
 'tops': [0],
 'nodes': [{'id': 0},
  {'id': 1, 'anchors': [{'from': 60, 'to': 63}]},
  {'id': 2, 'anchors': [{'from': 24, 'to': 29}]},
  {'id': 3, 'anchors': [{'from': 10, 'to': 23}]},
  {'id': 4, 'anchors': [{'from': 64, 'to': 117}]},
  {'id': 5, 'anchors': [{'from': 35, 'to': 59}]}],
 'edges': [{'id': 0, 'source': 0, 'target': 1, 'label': 'Attack'},
  {'id': 1, 'source': 1, 'target': 2, 'label': 'Time-Within'},
  {'id': 2, 'source': 1, 'target': 3, 'label': 'Place'},
  {'id': 3, 'source': 1, 'target': 4, 'label': 'Target'},
  {'id': 4, 'source': 1, 'target': 5, 'label': 'Instrument'}]}
```



### :feet: &nbsp; To train

To train EventGraph on different datasets, specify the configurations located in `perin/config`


```shell
cd perin

sbatch run.sh <config.yaml>
```


## Citation

```
TBA
```