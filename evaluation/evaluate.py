#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function
import sys
import os
import json

from nltk.tokenize.simple import SpaceTokenizer

tk = SpaceTokenizer()


def span_overlap(pred_span, gold_span):
    """
    calculate the overlap between a predicted argument span and a gold argument span
    """
    overlap = len(pred_span.intersection(gold_span)) / len(gold_span)
    return overlap
    

def convert_char_offsets_to_token_idxs(char_offsets, token_offsets):
    """
    char_offsets: list of str
    token_offsets: list of tuples

    >>> text = "I think the new uni ( ) is a great idea"
    >>> char_offsets = ["8:19"]
    >>> token_offsets =
    [(0,1), (2,7), (8,11), (12,15), (16,19), (20,21), (22,23), (24,26), (27,28), (29,34), (35,39)]

    >>> convert_char_offsets_to_token_idxs(char_offsets, token_offsets)
    >>> (2,3,4)
    """
    token_idxs = []
    #
    for char_offset in char_offsets:
        bidx, eidx = char_offset.split(":")
        bidx, eidx = int(bidx), int(eidx)
        intoken = False
        for i, (b, e) in enumerate(token_offsets):
            if b == bidx:
                intoken = True
            if intoken:
                token_idxs.append(i)
            if e == eidx:
                intoken = False
    return frozenset(token_idxs)


def convert_event_to_tuple(sentence):
    """
    >>> sentence 
    {'sent_id': 'nw/APW_ENG_20030322.0119/001',
    'text': 'U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn',
    'events': [
        {
            'event_type': 'Attack',
            'trigger': [['pounded'], ['121:128']],
            'arguments': [[['Baghdad'], ['129:136'], 'Place'], [['dawn'], ['140:144'], 'Time-Starting']]},
        {
            'event_type': 'Transport',
            'trigger': [['moving'], ['29:35']],
            'arguments': [[['Saturday'], ['81:89'], 'Time-Within'], [['the strategic southern port city of Basra'], ['39:80'], 'Destination'], [['U.S. and British troops'], ['0:23'], 'Artifact']]}
        
                ]
    
    }

    >>> event_tupels 
    [
        ((frozenset({20}), 'attack'), (frozenset({21}), 'place'), (frozenset({23}), 'time-starting')),
        ((frozenset({5}), 'transport'), (frozenset({14}), 'time-within'), (frozenset({7, 8, 9, 10, 11, 12, 13}), 'destination'), (frozenset({0, 1, 2, 3}), 'artifact'))
    ]
    
    """


    text = sentence['text']
    events = sentence['events']
    event_tuples = []
    token_offsets = list(tk.span_tokenize(text))

    if len(events) > 0:
        for event in events:
            event_tuple = tuple()

            trigger_char_idxs = event['trigger'][1]
            trigger = convert_char_offsets_to_token_idxs(trigger_char_idxs, token_offsets)
            event_type = event['event_type'].lower() if event['event_type'] else "none"

            event_tuple += ((trigger, event_type),)

            if len(event['arguments']) > 0:
                for argument in event['arguments']:
                    arg_role = argument[-1]
                    argument_char_idxs = argument[1]
                    arg = convert_char_offsets_to_token_idxs(argument_char_idxs, token_offsets)
                    

                    event_tuple += ((arg, arg_role),)
            
            event_tuples.append(event_tuple)
    return event_tuples        


def trigger_tuple_in_list(trigger_tuple, event_tuple_list, classification=False):
    if classification:
        for event_tuple in event_tuple_list:
            if trigger_tuple == event_tuple[0]:
                return True
        return False
    else:
        for event_tuple in event_tuple_list:
            if trigger_tuple[0] == event_tuple[0][0]:
                return True
        return False


def arg_tuple_in_tuple(arg_tuple, event_tuple, classification=False):
    for gold_arg in event_tuple[1:]:
        if arg_tuple[0] == gold_arg[0]:
            if classification:
                if arg_tuple[1] == gold_arg[1]:
                    return True
            else:
                return True 
    return False

def arg_tuple_overlap_tuple(arg_tuple, event_tuple, classification=False, overlap=0.75):
    """
    arg_tuple: predicted argument
    event_tunple: gold event tuple, with trigger followed by arguments

    overlap: the overlap ratio of a predicted argument with a gold argument
    """

    for gold_arg in event_tuple[1:]:
        if span_overlap(arg_tuple[0], gold_arg[0]) >= overlap:
            if classification:
                if arg_tuple[1] == gold_arg[1]:
                    return True
            else:
                return True
    return False

def trigger_precision(gold, pred, classification=False):
    tp = []
    fp = []

    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in ptuples:
            if trigger_tuple_in_list(stuple[0], gtuples, classification=classification):
                tp.append(1)
            else:
                fp.append(1)
    return sum(tp) / (sum(tp) + sum(fp) + 0.0000000000000001)


def trigger_recall(gold, pred, classification=False):
    tp = []
    fn = []

    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in gtuples:
            if trigger_tuple_in_list(stuple[0], ptuples, classification=classification):
                tp.append(1)
            else:
                fn.append(1)

    return sum(tp) / (sum(tp) + sum(fn) + 0.0000000000000001)


def trigger_f1(gold, pred, classification=False):
    precision = trigger_precision(gold, pred, classification=classification)
    recall = trigger_recall(gold, pred, classification=classification)

    f1 = 2 * (precision * recall) / (precision + recall + 0.00000000000000001)

    return precision, recall, f1


def argument_precision(gold, pred, classification=False):
    tp = []
    fp = []

    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in ptuples:
            if trigger_tuple_in_list(stuple[0], gtuples, classification=True):
                for g in gtuples:
                    if stuple[0] == g[0]:
                        gtuple = g
                for arg in stuple[1:]:
                    if arg_tuple_in_tuple(arg, gtuple, classification=classification):
                        tp.append(1)
                    else:
                        fp.append(1)
    return sum(tp) / (sum(tp) + sum(fp) + 0.0000000000000001)

def argument_span_precision(gold, pred, classification=False, overlap=0.75):
    tp = []
    fp = []

    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in ptuples:
            if trigger_tuple_in_list(stuple[0], gtuples, classification=True):
                for g in gtuples:
                    if stuple[0] == g[0]:
                        gtuple = g
                for arg in stuple[1:]:
                    if arg_tuple_overlap_tuple(arg, gtuple, classification=classification, overlap=overlap):
                        tp.append(1)
                    else:
                        fp.append(1)
    return sum(tp) / (sum(tp) + sum(fp) + 0.0000000000000001)


def argument_recall(gold, pred, classification=False):
    tp = []
    fn = []

    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in gtuples:
            if trigger_tuple_in_list(stuple[0], ptuples, classification=True):
                for p in ptuples:
                    if stuple[0] == p[0]:
                        ptuple = p
                for arg in stuple[1:]:
                    if arg_tuple_in_tuple(arg, ptuple, classification=classification):
                        tp.append(1)
                    else:
                        fn.append(1)
    return sum(tp) / (sum(tp) + sum(fn) + 0.0000000000000001)

def argument_span_recall(gold, pred, classification=False, overlap=0.75):
    tp = []
    fn = []

    for sent_idx in pred.keys():
        ptuples = pred[sent_idx]
        gtuples = gold[sent_idx]

        for stuple in gtuples:
            if trigger_tuple_in_list(stuple[0], ptuples, classification=True):
                for p in ptuples:
                    if stuple[0] == p[0]:
                        ptuple = p
                for arg in stuple[1:]:
                    if arg_tuple_overlap_tuple(arg, ptuple, classification=classification, overlap=overlap):
                        tp.append(1)
                    else:
                        fn.append(1)
    return sum(tp) / (sum(tp) + sum(fn) + 0.0000000000000001)


def argument_f1(gold, pred, classification=False):
    precision = argument_precision(gold, pred, classification=classification)
    recall = argument_recall(gold, pred, classification=classification)

    f1 = 2 * (precision * recall) / (precision + recall + 0.00000000000000001)

    return precision, recall, f1

def argument_span_f1(gold, pred, classification=False, overlap=0.75):
    precision = argument_span_precision(gold, pred, classification=classification, overlap=overlap)
    recall = argument_span_recall(gold, pred, classification=classification, overlap=overlap)

    f1 = 2 * (precision * recall) / (precision + recall + 0.00000000000000001)

    return precision, recall, f1