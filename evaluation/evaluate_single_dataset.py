import json 
from evaluate import convert_event_to_tuple, trigger_f1, argument_f1, argument_span_f1
import argparse


def evaluate(gold_file, pred_file):

    with open(gold_file) as f:
        gold = json.load(f)

    with open(pred_file) as f:
        preds = json.load(f)
    
    tgold = dict([(s["sent_id"], convert_event_to_tuple(s)) for s in gold])
    tpreds = dict([(s["sent_id"], convert_event_to_tuple(s)) for s in preds])

    g = sorted(tgold.keys())
    p = sorted(tpreds.keys())

    if g != p:
        print("Missing some sentences!")
        return 0.0, 0.0, 0.0
    
    trigger_idf = trigger_f1(tgold, tpreds, classification=False)
    trigger_cls = trigger_f1(tgold, tpreds, classification=True)



    argument_idf = argument_f1(tgold, tpreds, classification=False)
    argument_cls = argument_f1(tgold, tpreds, classification=True)

    results = {
        'trigger_identification': trigger_idf,
        'trigger_classification': trigger_cls,
        'argument_identification': argument_idf,
        'argument_classification': argument_cls
    }

    return results


def evaluate_span(gold_file, pred_file, overlap=0.75):

    with open(gold_file) as f:
        gold = json.load(f)

    with open(pred_file) as f:
        preds = json.load(f)
    
    tgold = dict([(s["sent_id"], convert_event_to_tuple(s)) for s in gold])
    tpreds = dict([(s["sent_id"], convert_event_to_tuple(s)) for s in preds])

    g = sorted(tgold.keys())
    p = sorted(tpreds.keys())

    if g != p:
        print("Missing some sentences!")
        return 0.0, 0.0, 0.0
    
    trigger_idf = trigger_f1(tgold, tpreds, classification=False)
    trigger_cls = trigger_f1(tgold, tpreds, classification=True)



    argument_idf = argument_span_f1(tgold, tpreds, classification=False, overlap=overlap)
    argument_cls = argument_span_f1(tgold, tpreds, classification=True, overlap=overlap)

    results = {
        'trigger_identification': trigger_idf,
        'trigger_classification': trigger_cls,
        'argument_identification': argument_idf,
        'argument_classification': argument_cls
    }

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gold_file", help="gold json file")
    parser.add_argument("pred_file", help="prediction json file")
    parser.add_argument("--span_overlap", help="argument overlap ratio", default=1, type=float)

    args = parser.parse_args()

    if args.span_overlap < 1:
        results = evaluate_span(args.gold_file, args.pred_file, overlap=args.span_overlap)
        print(f"Evaluate arguments with span overlap ratio of: {args.span_overlap}\n")
    else:
        results = evaluate(args.gold_file, args.pred_file)

    print(json.dumps(results, indent=2))
    print()
    print(list(results.values()))


if __name__ == "__main__":
    main()
