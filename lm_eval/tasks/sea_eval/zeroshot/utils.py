import tiger_eval


def tiger_eval_multichoice_question(items):
    data_with_model_prediction = []
    for item in items:
        parts = item[0].split("\n")

        new_sample = {}
        new_sample["choices"] = parts[:-1]
        new_sample["answer"] = parts[-1]
        new_sample["model_prediction"] = item[1]

        data_with_model_prediction.append(new_sample)

    for sample in data_with_model_prediction:
        sample["model_prediction_align"] = tiger_eval.multichoice_align.heuristic_align(
            sample["choices"],
            sample["model_prediction"],
        )
    accuracy = []
    for sample in data_with_model_prediction:
        accuracy.append(int(sample["model_prediction_align"] == sample["answer"]))

    return sum(accuracy) / len(accuracy)
