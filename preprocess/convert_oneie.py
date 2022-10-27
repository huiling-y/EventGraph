import json 
import argparse
import os


from extract_ace_events import read_file_to_list, dump_json, span_to_string


def load_json_lines(f_name):
    data = []
    with open(f_name, 'r') as f:
        for line in f.readlines():
            data.append(json.loads(line))
    return data

class OneieExtractor:
    def __init__(self, oneie_f, split_dir):
        self.ace_en = load_json_lines(oneie_f)
        self.data = []
        self.doc_ids = []
        self.train_split = read_file_to_list(f"{split_dir}/train.doc.txt")
        self.dev_split = read_file_to_list(f"{split_dir}/dev.doc.txt")
        self.test_split = read_file_to_list(f"{split_dir}/test.doc.txt")

    
    def retrieve_entity_span(self, argument, entities):
        """
        Get the exclusive token span of argument
        """
        for entity in entities:
            if entity['mention_id'] == argument['mention_id']:
                return entity['start'], entity['end']

    def new_span(self, lens, start, end):
        # update the span from token idx (inclusive)  to char idx
        # lens: list of length of each token
        new_start = sum(lens[:start]) + start
        new_end = new_start + sum(lens[start:end+1]) + (end-start)
        return new_start, new_end

    def get_doc_events(self, one_doc):

        sentences = one_doc['sentences']
        doc_id = one_doc['doc_id']


        for sent in sentences:
            if len(sent['tokens']) > 0:
                
                entities = sent['entities']

                sample = {}
                sample['sent_id'] = sent['sent_id']
                sample['text'] = ' '.join(sent['tokens'])
                sample['events'] = []

                if len(sent['events']) > 0:

                    sent_toks = sent['tokens']
                    lens = [len(t) for t in sent_toks]

                    for e in sent['events']:

                        a_event = {}
                        a_event['event_type'] = e['event_subtype']

                        trg_s, trg_e = e['trigger']['start'], e['trigger']['end']
                        trg_text = sent_toks[trg_s:trg_e]
                        trg_start, trg_end = self.new_span(lens, trg_s, trg_e-1)
                        
                        a_event['trigger'] = [[trg_text], [span_to_string(trg_start, trg_end)]]
                        a_event['arguments'] = []

                        for arg in e['arguments']:
                            arg_s, arg_e = self.retrieve_entity_span(arg, entities)
                            arg_text = ' '.join(sent_toks[arg_s:arg_e])
                            arg_role = arg['role']

                            arg_start, arg_end = self.new_span(lens, arg_s, arg_e-1)

                            a_arg = [[arg_text], [span_to_string(arg_start, arg_end)], arg_role]
                            a_event['arguments'].append(a_arg)

                        sample['events'].append(a_event)
                
                self.data.append(sample)
                self.doc_ids.append(doc_id)
    
    def convert(self):
        for doc in self.ace_en:
            self.get_doc_events(doc)
    
    def split_data(self):
        train_idx = [i for i,doc_id in enumerate(self.doc_ids) if doc_id in self.train_split]
        dev_idx = [i for i,doc_id in enumerate(self.doc_ids) if doc_id in self.dev_split]
        test_idx = [i for i,doc_id in enumerate(self.doc_ids) if doc_id in self.test_split]

        self.train = [self.data[i] for i in train_idx]
        self.dev = [self.data[i] for i in dev_idx]
        self.test = [self.data[i] for i in test_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--oneie_f', help='processed ace english events with OneIE')
    parser.add_argument('--split_dir', default='../dataset/splits2')
    parser.add_argument('--out_dir', default="../dataset/raw", help="directory processed dataset")

    args = parser.parse_args()

    # create subdir for ace_e_p
    os.makedirs(f"{args.out_dir}/ace_p_en", exist_ok=True)

    # convert ONEIE ace event data

    oneie_extractor = OneieExtractor(args.oneie_f, args.split_dir)
    oneie_extractor.convert()
    oneie_extractor.split_data()

    train_set, dev_set, test_set = oneie_extractor.train, oneie_extractor.dev, oneie_extractor.test

    # save converted data

    dump_json(f'{args.out_dir}/ace_p_en/train.json', train_set, indent=None)
    dump_json(f'{args.out_dir}/ace_p_en/dev.json', dev_set, indent=None)
    dump_json(f'{args.out_dir}/ace_p_en/test.json', test_set, indent=None)






