#!/usr/bin/env python3
# coding=utf-8

import utility.parser_utils as utils
from data.parser.from_mrp.abstract_parser import AbstractParser


class RequestParser(AbstractParser):
    def __init__(self, sentences, args, language: str, fields):
        self.data = {i: {"id": str(i), "sentence": sentence} for i, sentence in enumerate(sentences)}

        sentences = [example["sentence"] for example in self.data.values()]
    
        for example in zip(self.data.values()):
            example["input"] = example["input"].strip().split(' ')
            utils.create_token_anchors(example)

        for example in self.data.values():
            example["token anchors"] = [[a["from"], a["to"]] for a in example["token anchors"]]

        utils.create_bert_tokens(self.data, args.encoder)

        super(RequestParser, self).__init__(fields, self.data)
