#!/bin/bash

#indata -> input json file to be converted
#outdata -> output mrp file of converted graphs
# $1 -> dataset: ace_en, ace_p_en, ace_pp_en, ace_ppp_en


for split in train test dev; do
    indata=dataset/raw/"$1"/"$split".json
    outdata=dataset/labeled_edge_mrp/"$1"/"$split".mrp

    python mtool/main.py --strings --ids --read ace --write mrp "$indata" "$outdata"
done;