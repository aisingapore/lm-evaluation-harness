import datasets


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "id": doc["id"],
            "query": f"Question: {doc['question']}\nAnswer:",
            "choices": doc["choices"],
            "gold": ["A", "B", "C", "D", "E"].index(doc["answerKey"]),
        }
        return out_doc
    return dataset.map(_process_doc)
