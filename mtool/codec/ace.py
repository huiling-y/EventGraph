import json 
import sys 

from graph import Graph

def read(fp, text=None):
    def anchor(node):
        anchors = list()
        for string in node[1]:
            string = string.split(":")
            anchors.append({"from": int(string[0]), "to": int(string[1])})
        return anchors
    
    for native in json.load(fp):
        map = dict()

        try:
            graph = Graph(native["sent_id"], flavor=1, framework="ace")
            graph.add_input(native["text"])

            top = graph.add_node(top=True)

            for event in native['events']:

                trigger = event["trigger"]

                key = tuple(event['trigger'][1])
                if key in map:
                    trigger = map[key]
                else:
                    trigger = graph.add_node(
                        anchors=anchor(trigger)
                    )
                    map[key] = trigger 
                
                graph.add_edge(top.id, trigger.id, event["event_type"])


                arguments = event["arguments"]

                if len(arguments):
                    for argument in arguments:
                        arg_role = argument[-1]
                        key = tuple(argument[1])
                        if key in map:
                            argument = map[key]
                        else:
                            argument = graph.add_node(
                                anchors=anchor(argument)
                            )
                            map[key] = argument
                        
                        graph.add_edge(trigger.id, argument.id, arg_role)
            yield graph, None
        
        except Exception as error:
            print(
                f"codec.ace.read(): ignoring {native}: {error}",
                file=sys.stderr
            )


def get_text_span(node, text):
    anchored_text = [text[anchor['from']:anchor['to']] for anchor in node.anchors]
    anchors = [f"{anchor['from']}:{anchor['to']}" for anchor in node.anchors]
    return anchored_text, anchors



def write(graph, input):
    try:
        return write_labeled_edge(graph, input)
    except Exception as error:
        print(f"Problem with decoding sentence {graph.id}")
        raise error


def write_labeled_edge(graph, input):

    nodes = {node.id: node for node in graph.nodes}

    # create events
    events = {}
    for edge in graph.edges:

        if edge.src == 0:
            node = nodes[edge.tgt]
            events[node.id] = {
                'event_type': edge.lab,
                'trigger': [*get_text_span(node, input)],
                'arguments': []
            }
    
    # add event arguments
    for edge in graph.edges:
        if edge.src != 0:

            node = nodes[edge.tgt]
            anchored_text, anchors = get_text_span(node, input)

            events[edge.src]['arguments'].append([anchored_text, anchors, edge.lab])
    
    sentence = {
        "sent_id": graph.id,
        "text": input,
        "events": list(events.values())
    }
    return sentence

