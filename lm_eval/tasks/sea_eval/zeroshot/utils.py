import tiger_eval


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
