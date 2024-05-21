import random

import datasets


FEWSHOT_PROMPT = "Source Text:\n{}\n\nTranslation in English:\n{}\n\n"
QUERY_PROMPT = "Source Text:\n{}\n\nTranslation in English:\n"


def _process_doc(doc):
    queries = []
    n = len(doc["context"])
    random.seed(1234)
    for i in range(n):
        indices = random.sample(range(n), 6)

        count = 0
        query = ""
        for idx in indices:
            if doc["context"][idx] != doc["context"][i]:
                query += FEWSHOT_PROMPT.format(doc["context"][idx], doc["answer"][idx])
                count += 1
            if count == 5:
                break
        query += QUERY_PROMPT.format(doc["context"][i])
        queries.append(query)

    return {"query": queries}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(_process_doc, batched=True)

