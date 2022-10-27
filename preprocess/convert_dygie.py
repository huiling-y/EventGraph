import argparse
import os
from collections import defaultdict

from extract_ace_events import dump_json, span_to_string
from convert_oneie import load_json_lines

class DygieExtractor:
    def __init__(self, data_dir):
        self.ace_train = load_json_lines(f"{data_dir}/train.json")
        self.ace_dev = load_json_lines(f"{data_dir}/dev.json")
        self.ace_test = load_json_lines(f"{data_dir}/test.json")

    def new_span(self, lens, start, end):
        # update the span from token idx (inclusive)  to char idx
        # lens: list of length of each token
        new_start = sum(lens[:start]) + start
        new_end = new_start + sum(lens[start:end+1]) + (end-start)
        return new_start, new_end

    def get_doc_event(self, one_doc, data):
        file = one_doc['doc_key']
        sentences = one_doc['sentences']
        events = one_doc['events']
        sent_starts = one_doc['_sentence_start']
        entities = one_doc['ner']

        for i in range(len(sentences)):
            if len(events[i]) > 0:
                sent_start = sent_starts[i]

                sample = {}
                sample['file'] = file
                sample['text'] = ' '.join(sentences[i])
                sample['events'] = []

                sent_toks = sentences[i]
                lens = [len(t) for t in sent_toks]

                for j in range(len(events[i])):
                    a_event = {}
                    a_event['event_type'] = events[i][j][0][-1].split('.')[-1]
                    trg_text = sentences[i][events[i][j][0][0]-sent_start]
                    trg_s = trg_e = events[i][j][0][0]-sent_start # trigger token offset, inclusive span
                    trg_start, trg_end = self.new_span(lens, trg_s, trg_e)

                    a_event['trigger'] = [[trg_text], [span_to_string(trg_start, trg_end)]]
                    a_event['arguments'] = []

                    for arg in events[i][j][1:]:
                        arg_s, arg_e = arg[0]-sent_start, arg[1]-sent_start
                        arg_text = ' '.join(sent_toks[arg_s:arg_e+1])
                        arg_role = arg[-1]

                        arg_start, arg_end = self.new_span(lens, arg_s, arg_e)

                        a_arg = [[arg_text], [span_to_string(arg_start, arg_end)], arg_role]
                        a_event['arguments'].append(a_arg)
                    
                    sample['events'].append(a_event)
                
                data.append(sample)

            else:
                if len(sentences[i]) > 0:
                    sample = {}
                    sample['file'] = file
                    sample['text'] = ' '.join(sentences[i])
                    sample['events'] = []

                    data.append(sample)
    
    def get_events(self, data):
        events = []
        for doc in data:
            self.get_doc_event(doc, events)
        return events

    def update_sent_ids(self, data):
        new_data = []
        files = [d['file'] for d in data]
        
        # map file to idx 
        map_file_idx = defaultdict(list)
        for i,f in enumerate(files):
            map_file_idx[f].append(i)
        
        # obtain sentence ids -> "file/sent_num"
        sent_ids = ['null'] * len(files)
        for k,v in map_file_idx.items():
            for j in range(len(v)):
                sent_ids[v[j]] = f"{k}/{j+1:03d}"
        
        # update sent ids
        for m in range(len(data)):
            sample = data[m]
            sample['sent_id'] = sent_ids[m]
            sample.pop('file')
            new_data.append(sample)
        return new_data
    
    def convert(self):
        train_events = self.get_events(self.ace_train)
        dev_events = self.get_events(self.ace_dev)
        test_events = self.get_events(self.ace_test)

        self.train = self.update_sent_ids(train_events)
        self.dev = self.update_sent_ids(dev_events)
        self.test = self.update_sent_ids(test_events)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dygie_data_dir', help='directory of processed ace english events with dygie')
    parser.add_argument('--out_dir', default="../dataset/raw", help="directory processed dataset")

    args = parser.parse_args()

    # create subdir for ace_e_p
    os.makedirs(f"{args.out_dir}/ace_en", exist_ok=True)

    # convert DYGIE data

    dygie_extractor = DygieExtractor(args.dygie_data_dir)
    dygie_extractor.convert()


    train_set, dev_set, test_set = dygie_extractor.train, dygie_extractor.dev, dygie_extractor.test

    # save converted data

    dump_json(f'{args.out_dir}/ace_en/train.json', train_set, indent=None)
    dump_json(f'{args.out_dir}/ace_en/dev.json', dev_set, indent=None)
    dump_json(f'{args.out_dir}/ace_en/test.json', test_set, indent=None)



