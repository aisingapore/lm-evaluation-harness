import datasets
import numpy as np

def indommlu_doc_to_target(doc):
    processed_options = list(map(lambda x: x[0], doc["options"]))
    return processed_options.index(doc["answer"])

def indommlu_doc_to_choice(doc):
    return list(map(lambda x: x.split(' ', 1)[1], eval(doc["options"])))

def _process_doc(dataset: datasets.Dataset):
    def _helper(doc):
        options = eval(doc["options"])
        options_string = '\n'.join(options)
        idx = ord(doc["answer"]) - ord("A")
        if idx >= len(options):
            idx = len(options) - 1
            doc["to_compile"] = 0
        else:
            doc["to_compile"] = 1
        doc["prompt"] = f"Ini adalah soal {doc['subject']} untuk {doc['level']}. Pilihlah salah satu jawaban yang dianggap benar!\n\n{doc['question']}\n{options_string}\n\nJawaban:"
        doc["doc_to_target"] = idx
        doc["doc_to_choice"] = options
        return doc
    return dataset.map(_helper)

def _process_results(doc, results):
    result_dict = {}
    lls, is_greedy = zip(*results)

    # retrieve choices in List[str] form, to compute choice lengths, etc.
    choices = doc["doc_to_choice"]
    completion_len = np.array([float(len(i)) for i in choices])

    pred = np.argmax(lls)
    pred_norm = np.argmax(lls / completion_len)

    gold = doc["doc_to_target"]
    gold_index_error = False
    if isinstance(gold, list):
        gold = [i if i < len(choices) else -100 for i in gold]
        if -100 in gold:
            gold_index_error = True
    else:
        if isinstance(gold, int):
            gold = gold if gold < len(choices) else -100
        elif isinstance(gold, str):
            gold = choices.index(gold) if gold in choices else -100

        if gold == -100:
            gold_index_error = True

    if gold_index_error:
        eval_logger.warning(
            f"Label index was not in within range of available choices,"
            f"Sample:\n\n{doc}\n\n"
        )

    acc = 1.0 if pred == gold else 0.0
    acc_norm = 1.0 if pred_norm == gold else 0.0
    # TODO: this gets score of 0 on arc_challenge for pythia-70m. need to test that this works properly
    exact_match = int(is_greedy[gold]) if gold != -100 else 0

    acc *= doc["to_compile"]
    acc_norm *= doc["to_compile"]
    exact_match *= doc["to_compile"]

    result_dict = {
        **({"acc": acc}),
        **({"acc_norm": acc_norm})
    }

    return result_dict
