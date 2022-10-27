#!/usr/bin/env python3
# coding=utf-8

from data.parser.to_mrp.abstract_parser import AbstractParser

# Full argument set

pairs_full = {
    "Attack": ['Instrument', 'Time-Before', 'Victim', 'Time-At-End', 'Time-After', 'Time-Starting', 'Place', 'Agent', 'Target', 'Time-Within', 'Attacker', 'Time-Ending', 'Time-At-Beginning', 'Time-Holds'],
    "Transport": ['Artifact', 'Vehicle', 'Victim', 'Time-Before', 'Origin', 'Time-At-End', 'Destination', 'Time-After', 'Time-Starting', 'Agent', 'Place', 'Time-Within', 'Time-Ending', 'Time-At-Beginning', 'Time-Holds'],
    "Die": ['Instrument', 'Victim', 'Time-Before', 'Time-After', 'Time-Starting', 'Agent', 'Place', 'Time-Within', 'Time-Ending', 'Person', 'Time-At-Beginning', 'Time-Holds'],
    "Meet": ['Entity', 'Time-After', 'Time-Starting', 'Place', 'Time-Within', 'Time-Ending', 'Time-At-Beginning', 'Time-Holds'],
    "End-Position": ['Entity', 'Time-Before', 'Time-At-End', 'Position', 'Time-After', 'Time-Starting', 'Place', 'Time-Within', 'Time-Ending', 'Person', 'Time-Holds'],
    "Transfer-Money": ['Time-Before', 'Time-After', 'Recipient', 'Money', 'Giver', 'Time-Starting', 'Place', 'Time-Within', 'Beneficiary', 'Time-Holds'],
    "Elect": ['Entity', 'Time-Before', 'Position', 'Time-Starting', 'Place', 'Time-Within', 'Person', 'Time-At-Beginning', 'Time-Holds'],
    "Injure": ['Victim', 'Instrument', 'Agent', 'Place', 'Time-Within'],
    "Transfer-Ownership": ['Artifact', 'Time-Before', 'Time-Ending', 'Buyer', 'Place', 'Time-Within', 'Seller', 'Price', 'Beneficiary', 'Time-At-Beginning'],
    "Phone-Write": ['Entity', 'Time-Before', 'Time-After', 'Time-Starting', 'Place', 'Time-Within', 'Time-Holds'],
    "Start-Position": ['Entity', 'Time-Before', 'Position', 'Time-After', 'Time-Starting', 'Place', 'Time-Within', 'Person', 'Time-At-Beginning', 'Time-Holds'],
    "Trial-Hearing": ['Defendant', 'Crime', 'Time-At-End', 'Time-Starting', 'Prosecutor', 'Place', 'Time-Within', 'Adjudicator', 'Time-Holds'],
    "Charge-Indict": ['Defendant', 'Crime', 'Time-Before', 'Prosecutor', 'Place', 'Time-Within', 'Adjudicator', 'Time-Ending'],
    "Sentence": ['Defendant', 'Crime', 'Time-At-End', 'Time-Starting', 'Place', 'Time-Within', 'Sentence', 'Adjudicator'],
    "Arrest-Jail": ['Crime', 'Time-Before', 'Time-Starting', 'Agent', 'Place', 'Time-Within', 'Time-Ending', 'Person', 'Time-At-Beginning', 'Time-Holds'],
    "Marry": ['Time-Before', 'Place', 'Time-Within', 'Person', 'Time-Holds'],
    "Demonstrate": ['Entity', 'Time-At-End', 'Time-Starting', 'Place', 'Time-Within'],
    "Sue": ['Defendant', 'Crime', 'Plaintiff', 'Place', 'Time-Within', 'Adjudicator', 'Time-Holds'],
    "Convict": ['Defendant', 'Crime', 'Place', 'Time-Within', 'Adjudicator', 'Time-At-Beginning'],
    "Be-Born": ['Place', 'Time-Within', 'Person', 'Time-Holds'],
    "Start-Org": ['Time-Before', 'Org', 'Time-After', 'Time-Starting', 'Agent', 'Place', 'Time-Within'],
    "Release-Parole": ['Entity', 'Crime', 'Time-After', 'Place', 'Time-Within', 'Person'],
    "Declare-Bankruptcy": ['Org', 'Time-After', 'Place', 'Time-Within', 'Time-At-Beginning'],
    "Appeal": ['Crime', 'Plaintiff', 'Place', 'Time-Within', 'Adjudicator', 'Time-Holds'],
    "End-Org": ['Org', 'Time-After', 'Place', 'Time-Within', 'Time-At-Beginning', 'Time-Holds'],
    "Divorce": ['Place', 'Time-Within', 'Person'],
    "Fine": ['Entity', 'Crime', 'Money', 'Place', 'Time-Within', 'Adjudicator'],
    "Execute": ['Crime', 'Time-After', 'Agent', 'Place', 'Time-Within', 'Person', 'Time-At-Beginning'],
    "Merge-Org": ['Time-Ending', 'Org'],
    "Nominate": ['Agent', 'Position', 'Time-Within', 'Person'],
    "Extradite": ['Origin', 'Destination', 'Agent', 'Time-Within', 'Person'],
    "Acquit": ['Time-Within', 'Defendant', 'Adjudicator', 'Crime'],
    "Pardon": ['Place', 'Defendant', 'Time-At-End', 'Adjudicator']
}

# Time and value arguments excluded

pairs_short = {
    'Attack': ['Instrument', 'Victim', 'Place', 'Agent', 'Target', 'Attacker'],
    'Transport': ['Artifact', 'Vehicle', 'Victim', 'Origin', 'Destination', 'Agent', 'Place'],
    'Die': ['Instrument', 'Victim', 'Agent', 'Place', 'Person'],
    'Meet': ['Entity', 'Place'],
    'End-Position': ['Entity', 'Place', 'Person'],
    'Transfer-Money': ['Recipient', 'Giver', 'Place', 'Beneficiary'],
    'Elect': ['Entity', 'Place', 'Person'],
    'Injure': ['Victim', 'Instrument', 'Agent', 'Place'],
    'Transfer-Ownership': ['Artifact', 'Buyer', 'Place', 'Seller', 'Beneficiary'],
    'Phone-Write': ['Entity', 'Place'],
    'Start-Position': ['Entity', 'Place', 'Person'],
    'Trial-Hearing': ['Defendant', 'Prosecutor', 'Place', 'Adjudicator'],
    'Charge-Indict': ['Defendant', 'Prosecutor', 'Place', 'Adjudicator'],
    'Sentence': ['Defendant', 'Place', 'Adjudicator'],
    'Arrest-Jail': ['Agent', 'Place', 'Person'],
    'Marry': ['Place', 'Person'],
    'Demonstrate': ['Entity', 'Place'],
    'Sue': ['Defendant', 'Plaintiff', 'Place', 'Adjudicator'],
    'Convict': ['Defendant', 'Place', 'Adjudicator'],
    'Be-Born': ['Place', 'Person'],
    'Start-Org': ['Org', 'Agent', 'Place'],
    'Release-Parole': ['Entity', 'Place', 'Person'],
    'Declare-Bankruptcy': ['Org', 'Place'],
    'Appeal': ['Plaintiff', 'Place', 'Adjudicator'],
    'End-Org': ['Org', 'Place'],
    'Divorce': ['Place', 'Person'],
    'Fine': ['Entity', 'Place', 'Adjudicator'],
    'Execute': ['Agent', 'Place', 'Person'],
    'Merge-Org': ['Org'],
    'Nominate': ['Agent', 'Person'],
    'Extradite': ['Origin', 'Destination', 'Agent', 'Person'],
    'Acquit': ['Defendant', 'Adjudicator'],
    'Pardon': ['Place', 'Defendant', 'Adjudicator']
}



class LabeledEdgeParser(AbstractParser):
    def __init__(self, *args):
        super().__init__(*args)

        self.event_types = ['Be-Born', 'Die', 'Marry', 'Divorce', 'Injure', 'Transfer-Ownership', 'Transfer-Money', 'Transport', 'Start-Org', \
            'End-Org', 'Declare-Bankruptcy', 'Merge-Org', 'Attack', 'Demonstrate', 'Meet', 'Phone-Write', 'Start-Position', 'End-Position', 'Nominate', \
                'Elect', 'Arrest-Jail', 'Release-Parole', 'Charge-Indict', 'Trial-Hearing', 'Sue', 'Convict', 'Sentence', 'Fine', 'Execute', 'Extradite', 'Acquit', 'Pardon', 'Appeal']
        
        if len(self.dataset.edge_label_field.vocab) == 67:
            self.argument_roles = ['Person', 'Place', 'Buyer', 'Seller', 'Beneficiary', 'Price', 'Artifact', 'Origin', 'Destination', 'Giver', \
                'Recipient', 'Money', 'Org', 'Agent', 'Victim', 'Instrument', 'Entity', 'Attacker', 'Target', 'Defendant', 'Adjudicator', 'Prosecutor', \
                    'Plaintiff', 'Crime', 'Position', 'Vehicle', 'Time-Within', 'Time-Starting', 'Time-Ending', 'Time-Before', 'Time-After', 'Time-Holds', 'Time-At-Beginning', 'Time-At-End']
            pairs = pairs_full
        else:
            self.argument_roles = ['Instrument', 'Artifact', 'Buyer', 'Person', 'Entity', 'Seller', 'Vehicle', 'Victim', 'Attacker', 'Recipient', 'Adjudicator', 'Giver', 'Origin', \
                'Beneficiary', 'Destination', 'Prosecutor', 'Plaintiff', 'Defendant', 'Org', 'Place', 'Target', 'Agent']
            pairs = pairs_short

        self.pairs = {}
        for k,v in pairs.items():
            self.pairs[k] = [self.dataset.edge_label_field.vocab.stoi[item] for item in v]
        
        self.argument_ids = [self.dataset.edge_label_field.vocab.stoi[arg] for arg in self.argument_roles]
        self.event_type_ids = [self.dataset.edge_label_field.vocab.stoi[e] for e in self.event_types]

    def parse(self, prediction):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        output["nodes"] = self.create_anchors(prediction, output["nodes"], join_contiguous=True, at_least_one=True)
        output["nodes"] = [{"id": 0}] + output["nodes"]
        output["edges"] = self.create_edges(prediction, output["nodes"])

        return output

    def create_nodes(self, prediction):
        return [{"id": i + 1} for i, l in enumerate(prediction["labels"])]

    def create_edges(self, prediction, nodes):
        N = len(nodes)
        edge_prediction = prediction["edge presence"][:N, :N]

        edges = []
        event_nodes = []
        event_t = []
        for target in range(1, N):
            if edge_prediction[0, target] >= 0.5:
                for j in self.argument_ids:
                    prediction['edge labels'][0, target, j] = float('-inf')
                self.create_edge(0, target, prediction, edges, nodes)
                event_nodes.append(target)
                event_t.append(edges[-1]['label'])                

        for source in range(1, N):
            for target in range(1, N):
                if source == target:
                    continue
                if edge_prediction[source, target] < 0.5:
                    continue

                if source in event_nodes:
                    per_etype = event_t[event_nodes.index(source)]
                    candidates = self.pairs[per_etype]
                    for j in range(len(self.dataset.edge_label_field.vocab)):
                        if j not in candidates:
                            prediction['edge labels'][source, target, j] = float('-inf')

                    self.create_edge(source, target, prediction, edges, nodes)

        return edges

    def get_edge_label(self, prediction, source, target):
        return self.dataset.edge_label_field.vocab.itos[prediction["edge labels"][source, target].argmax(-1).item()]
