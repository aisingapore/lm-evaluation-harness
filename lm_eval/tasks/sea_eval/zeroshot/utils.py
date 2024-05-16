import tiger_eval


def rouge1(items):
    return items


def rouge2(items):
    return items


def rougeL(items):
    return items


def avg_rouge(items):
    return items


def tiger_eval_multichoice_question(items):
    data_with_model_prediction = []
    for item in items:
        parts = item[0].split("\n")

        new_sample = {}
        new_sample["choices"] = parts[:-1]
        new_sample["answer"] = parts[-1]
        new_sample["model_prediction"] = item[1]

        data_with_model_prediction.append(new_sample)

    return tiger_eval.multichoice_question.score(
        data_with_model_prediction, category=False
    )["accuracy"]


def tiger_eval_rouge1(items):
    return _tiger_eval_rouge(items)["rouge1"]


def tiger_eval_rouge2(items):
    return _tiger_eval_rouge(items)["rouge2"]


def tiger_eval_rougeL(items):
    return _tiger_eval_rouge(items)["rougeL"]


def tiger_eval_avg_rouge(items):
    return _tiger_eval_rouge(items)["avg_rouge"]


def tiger_eval_translation_bleu(items):
    data_with_model_prediction = []
    for item in items:
        new_sample = {}
        new_sample["answer"] = item[0]
        new_sample["model_prediction"] = item[1]

        data_with_model_prediction.append(new_sample)

    return tiger_eval.translation_bleu.score(
        data_with_model_prediction
    )["bleu_score"]


def _tiger_eval_rouge(items):
    data_with_model_prediction = []
    for item in items:
        new_sample = {}
        new_sample["answer"] = item[0]
        new_sample["model_prediction"] = item[1]

        data_with_model_prediction.append(new_sample)

    return tiger_eval.rouge.score(data_with_model_prediction)
