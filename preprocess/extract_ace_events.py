import json 
import os
from collections import defaultdict
from extractor import Extractor
import argparse

def read_file_to_list(f_name):
    with open(f_name, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_json(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def dump_json(file, data, indent=0):
    with open(file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def span_to_string(start, end):
    return f"{start}:{end}"

def string_to_span(span):
    span = span.split(':')
    start, end = int(span[0]), int(span[1])
    return start, end

class FilterArg:
    def __init__(self, train, dev, test):
        self.train = train 
        self.dev = dev
        self.test = test 
        self.filter_args = ['Time-Within', 'Crime', 'Money', 'Time-Holds', 'Sentence', 'Time-Starting', 'Time-Before', \
               'Time-After', 'Time-Ending', 'Time-At-Beginning', 'Time-At-End', 'Position','Price']
    
    def update_args(self, args):
        new_args = []
        for arg in args:
            if arg[-1] not in self.filter_args:
                new_args.append(arg)
        return new_args
    
    def update_dataset(self, data):
        for sent in data:
            for event in sent['events']:
                event['arguments'] = self.update_args(event['arguments'])

    def update(self):
        self.update_dataset(self.train)
        self.update_dataset(self.dev)
        self.update_dataset(self.test)



class EventProcessor:
    def __init__(self, events, non_events):
        self.events = events
        self.non_events = non_events
    
    def clean_arguments(self):
        # remove entities that are non-event arguments
        clean_events = self.events.copy()
        for e in clean_events:
            args = [arg for arg in e['entities'] if arg['role'] != 'None']
            e['arguments'] = args
        for e in clean_events:
            e.pop('entities')
        
        self.clean_events = clean_events

    def new_span(self, lens, start, end):
        # update the span from token idx (inclusive)  to char idx
        # lens: list of length of each token
        new_start = sum(lens[:start]) + start
        new_end = new_start + sum(lens[start:end+1]) + (end-start)
        return new_start, new_end

    def collapse_one(self, e):
        d = {}
        d['event_type'] = e['event_type']
        
        # create a list of token length for converting span offset
        lens = [len(t) for t in e['tokens']]
        text = " ".join(e['tokens'])

        trigger = []
        trigger.append([' '.join(e['trigger_tokens'])]) # trigger text
        trg_start, trg_end = self.new_span(lens, e['trigger_start'], e['trigger_end'])
        trigger.append([span_to_string(trg_start, trg_end)])
        d['trigger'] = trigger
        
        args = []
        
        for arg in e['arguments']:
            tokens = ' '.join(arg['tokens'])
            arg_start, arg_end = self.new_span(lens, arg['idx_start'], arg['idx_end'])
            span = span_to_string(arg_start, arg_end)
            role = arg['role']
            
            an_arg = [[tokens], [span], role]
            
            args.append(an_arg)
        
        d['arguments'] = args
        
        return d
    
    def merge_events(self):
        merged_events = []

        # map sentence to idx
        map_sent_idx = defaultdict(list)
        sents = [' '.join(e['tokens']) for e in self.clean_events]
        for i, sent in enumerate(sents):
            map_sent_idx[sent].append(i)
        
        # map "dir/file_name" to idx 
        files = []
        for k,v in map_sent_idx.items():
            files.append(self.clean_events[v[0]]['dir'] + '/' + self.clean_events[v[0]]['file'])

        map_fname_idx = defaultdict(list)
        for i, f in enumerate(files):
            map_fname_idx[f].append(i)
        
        self.event_map_fname_idx = map_fname_idx

        # obtain sentence ids -> dir/file_name/sent_number
        sent_ids = ['null'] * len(files)

        for k,v in map_fname_idx.items():
            for i in range(len(v)):
                sent_ids[v[i]] = f"{k}/{i+1:03d}"

        # merge events by sentence
        for i in range(len(sent_ids)):
            sample = {}
            sample['sent_id'] = sent_ids[i]
            sample['text'] = list(map_sent_idx.keys())[i]

            sample_events = []
            for j in list(map_sent_idx.values())[i]:
                sample_events.append(self.collapse_one(self.clean_events[j]))
            
            sample['events'] = sample_events

            merged_events.append(sample)
            
        self.processed_events = merged_events
    
    def merge_non_events(self):
        merged_non_events = []

        # map "dir/file_name" to idx
        map_fname_idx = defaultdict(list)
        for i,e in enumerate(self.non_events):
            map_fname_idx[e['dir'] + '/' + e['file']].append(i)
        
        # sentence id starting num
        start_idx = defaultdict()
        for k in map_fname_idx:
            start_idx[k] = len(self.event_map_fname_idx[k])
        
        # obtain sentence ids
        sent_ids = ['null'] * len(self.non_events)
        for k,v in map_fname_idx.items():
            for i in range(len(v)):
                sent_ids[v[i]] = f"{k}/{i+1+start_idx[k]:03d}"
            
        # merge non-event sents
        for i in range(len(sent_ids)):
            sample = {}
            sample['sent_id'] = sent_ids[i] 
            sample['text'] = ' '.join(self.non_events[i]['tokens'])
            sample['events'] = []
            merged_non_events.append(sample)
        
        self.processed_non_events = merged_non_events



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', help='ace english data: /ace_2005_td_v7/data/English')
    parser.add_argument('--corenlp_dir', help='corenlp dir: stanford-corenlp-full-2018-02-27')
    parser.add_argument('--split_dir', default='../dataset/splits')
    parser.add_argument('--out_dir', default="../dataset/raw")

    args = parser.parse_args()

    # creat subdir for ace_e_ppp and ace_e_pp
    os.makedirs(f"{args.out_dir}/ace_ppp_en", exist_ok=True)
    os.makedirs(f"{args.out_dir}/ace_pp_en", exist_ok=True)

    # extract raw events from ACE

    extractor = Extractor(args.data_dir, args.corenlp_dir)
    extractor.Files_Extract()
    extractor.Entity_Extract()
    extractor.Event_Extract()
    extractor.None_event_Extract()
    extractor.process()

    events, non_events = extractor.Events, extractor.None_events

    # process extracted events

    processor = EventProcessor(events, non_events)
    processor.clean_arguments()
    processor.merge_events()
    processor.merge_non_events()

    all_sample = processor.processed_events + processor.processed_non_events

    # split data

    train_split = read_file_to_list(f'{args.split_dir}/train.txt')
    dev_split = read_file_to_list(f'{args.split_dir}/dev.txt')
    test_split = read_file_to_list(f'{args.split_dir}/test.txt')

    train_set = [instance for instance in all_sample \
        if instance['sent_id'].split('/')[1].replace('.','_').replace('-','_') in train_split]

    dev_set = [instance for instance in all_sample \
        if instance['sent_id'].split('/')[1].replace('.','_').replace('-','_') in dev_split]

    test_set = [instance for instance in all_sample \
        if instance['sent_id'].split('/')[1].replace('.','_').replace('-','_') in test_split]

    # save raw data for ACE-E+++

    dump_json(f'{args.out_dir}/ace_ppp_en/train.json', train_set, indent=None)
    dump_json(f'{args.out_dir}/ace_ppp_en/dev.json', dev_set, indent=None)
    dump_json(f'{args.out_dir}/ace_ppp_en/test.json', test_set, indent=None)

    # Obtain and save data for ACE-E++

    arg_filter = FilterArg(train_set, dev_set, test_set)
    arg_filter.update()

    ace_e_pp_train, ace_e_pp_dev, ace_e_pp_test = arg_filter.train, arg_filter.dev, arg_filter.test 


    dump_json(f'{args.out_dir}/ace_pp_en/train.json', ace_e_pp_train, indent=None)
    dump_json(f'{args.out_dir}/ace_pp_en/dev.json', ace_e_pp_dev, indent=None)
    dump_json(f'{args.out_dir}/ace_pp_en/test.json', ace_e_pp_test, indent=None)


    













        

