import datasets

from lm_eval.api.samplers import ContextSampler

QUERY_PROMPT = "Question: {question}\nChoices:\n{choices}Answer:"
KEYS = ["A", "B", "C", "D"]


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "query": QUERY_PROMPT.format(
                question=doc["question"],
                choices="".join(
                    f"{key}. {choice}\n" for key, choice in zip(KEYS, doc["choices"])
                ),
            ),
            "choices": doc["choices"],
            "gold": (
                KEYS.index(doc["answer"]) if isinstance(doc["answer"], str) else doc["answer"]
            ),
        }
        return out_doc
    return dataset.map(_process_doc)

