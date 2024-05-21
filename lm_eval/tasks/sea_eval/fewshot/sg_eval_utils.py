import random

import datasets


FEWSHOT_PROMPT = "Question:\n{}\n\nChoices:\n{}\n\nAnswer:\n{}\n\n"
QUERY_PROMPT = "Question:\n{}\n\nChoices:\n{}\n\nAnswer:\n"


def _process_doc(doc):
    queries = []
    targets = []
    n = len(doc["question"])
    random.seed(1234)
    for i in range(n):
        indices = random.sample(range(n), 6)

        count = 0
        query = ""
        for idx in indices:
            if doc["question"][idx] != doc["question"][i]:
                query += FEWSHOT_PROMPT.format(
                    doc["question"][idx], "\n".join(doc["choices"][idx]), doc["answer"][idx]
                )
                count += 1
            if count == 5:
                break
        query += QUERY_PROMPT.format(doc["question"][i], "\n".join(doc["choices"][i]))
        queries.append(query)
        targets.append("{}\n{}".format("\n".join(doc["choices"][i]), doc["answer"][i]))

    return {"query": queries, "align_target": targets}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_doc, batched=True, remove_columns=dataset.column_names)
