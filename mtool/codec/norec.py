import json
import sys

from graph import Graph


def read(fp, text=None, node_centric=False):
    def anchor(node):
        anchors = list()
        for string in node[1]:
            string = string.split(":")
            anchors.append({"from": int(string[0]), "to": int(string[1])})
        return anchors

    for native in json.load(fp):
        map = dict()
        try:
            graph = Graph(native["sent_id"], flavor=1, framework="norec")
            graph.add_input(native["text"])

            if not node_centric:
                top = graph.add_node(top=True)

            for opinion in native["opinions"]:
                expression = opinion["Polar_expression"]
                properties, values = ["Intensity"], [opinion["Intensity"]]

                if node_centric:
                    expression = graph.add_node(
                        label=opinion["Polarity"],
                        top=True,
                        properties=properties,
                        values=values,
                        anchors=anchor(expression),
                    )
                else:
                    expression = graph.add_node(
                        properties=properties,
                        values=values,
                        anchors=anchor(expression),
                    )
                    key = tuple(opinion["Polar_expression"][1])
                    if key in map:
                        print("we got double expression here", native["sent_id"])
                    map[key] = expression

                    graph.add_edge(top.id, expression.id, opinion["Polarity"])

                source = opinion["Source"]
                if len(source[1]):
                    key = tuple(source[1])
                    if key in map:
                        source = map[key]
                    else:
                        source = graph.add_node(
                            label="Source" if node_centric else None,
                            anchors=anchor(source),
                        )
                        map[key] = source
                    graph.add_edge(expression.id, source.id, None if node_centric else "Source")

                target = opinion["Target"]
                if len(target[1]):
                    key = tuple(target[1])
                    if key in map:
                        target = map[key]
                    else:
                        target = graph.add_node(
                            label="Target" if node_centric else None,
                            anchors=anchor(target),
                        )
                        map[key] = target
                    graph.add_edge(expression.id, target.id, None if node_centric else "Target")

            yield graph, None

        except Exception as error:
            print(
                f"codec.norec.read(): ignoring {native}: {error}",
                file=sys.stderr,
            )


def get_text_span(node, text):
    anchored_text = [text[anchor['from']:anchor['to']] for anchor in node.anchors]
    anchors = [f"{anchor['from']}:{anchor['to']}" for anchor in node.anchors]
    return anchored_text, anchors


def write(graph, input, node_centric=False):
    try:
        if node_centric:
            return write_node_centric(graph, input)
        return write_labeled_edge(graph, input)

    except Exception as error:
        print(f"Problem with decoding sentence {graph.id}")
        raise error


def write_node_centric(graph, input):
    nodes = {node.id: node for node in graph.nodes}

    # create opinions
    opinions = {}
    for node in graph.nodes:
        if node.label in ["Source", "Target"]:
            continue
        opinions[node.id] = {
            "Source": [[], []],
            "Target": [[], []],
            "Polar_expression": [*get_text_span(node, input)],
            "Polarity": node.label,
        }
        if node.properties is not None and len(node.properties) > 0:
            for key, value in zip(node.properties, node.values):
                opinions[node.id][key] = value

    # add sources & targets
    for edge in graph.edges:
        if edge.src not in opinions:
            continue

        target_node = nodes[edge.tgt]
        if target_node.label not in ["Source", "Target"]:
            continue

        anchored_text, anchors = get_text_span(target_node, input)
        opinions[edge.src][target_node.label][0] += anchored_text
        opinions[edge.src][target_node.label][1] += anchors

    sentence = {
        "sent_id": graph.id,
        "text": input,
        "opinions": list(opinions.values()),
    }
    return sentence


def write_labeled_edge(graph, input):
    nodes = {node.id: node for node in graph.nodes}

    # create opinions
    opinions = {}
    for edge in graph.edges:
        if edge.lab in ["Source", "Target"]:
            continue

        node = nodes[edge.tgt]
        opinions[node.id] = {
            "Source": [[], []],
            "Target": [[], []],
            "Polar_expression": [*get_text_span(node, input)],
            "Polarity": edge.lab,
        }

    # add sources & targets
    for edge in graph.edges:
        if edge.lab not in ["Source", "Target"]:
            continue
        if edge.src not in opinions:
            continue

        node = nodes[edge.tgt]
        anchored_text, anchors = get_text_span(node, input)

        opinions[edge.src][edge.lab][0] += anchored_text
        opinions[edge.src][edge.lab][1] += anchors

    sentence = {
        "sent_id": graph.id,
        "text": input,
        "opinions": list(opinions.values()),
    }
    return sentence
