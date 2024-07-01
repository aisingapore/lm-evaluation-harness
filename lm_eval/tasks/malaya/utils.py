import functools
import random
import re
from typing import Iterable, List, Tuple, Union

import datasets

from lm_eval.api.filter import Filter

FEWSHOT_PROMPT = "Contoh soalan {index}\nobjektif: {instruction}\nsoalan: {choices}\njawapan: {answer}"
QUERY_PROMPT = "objektif: {instruction}\nsoalan: {choices}\njawapan:"


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        out_doc = {
            "query": QUERY_PROMPT.format(
                instruction=doc["objektif"], choices=doc["soalan"]
            ),
            "target": [doc["jawapan"]]
        }
        return out_doc
    return dataset.map(_process_doc)


def process_docs_1shot(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc_fewshot, num_fewshot=1)
    return dataset.map(
        process_doc_partial, batched=True, remove_columns=dataset.column_names
    )


def process_docs_3shot(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc_fewshot, num_fewshot=3)
    return dataset.map(
        process_doc_partial, batched=True, remove_columns=dataset.column_names
    )


def most_common(arr):
    return max(set(arr), key=arr.count)


def acc(items):
    return int(most_common(items[1]) == items[0])


class MultiChoiceFilter:
    def apply(self, resps: Union[List, Iterable], docs: List[dict]) -> Iterable:
        def extract_choice(resp):
            extracted_resps = []
            for response in resp:
                extracted = response.split("jawapan:")[-1].strip()
                if extracted:
                    extracted = extracted.split()[0]
                    extracted = MultiChoiceFilter._multi_replace(
                        extracted, (".", "</s>"), ""
                    ).split("\\")[0].split("/")[0]
                extracted_resps.append(extracted)
            return extracted_resps
        return map(extract_choice, resps)


    @staticmethod
    def _multi_replace(string: str, old_values: Tuple[str, ...], new_value: str):
        if not old_values:
            return string

        sorted_old_values = sorted(old_values, key=len, reverse=True)
        escaped_old_values = map(re.escape, sorted_old_values)
        pattern = re.compile("|".join(escaped_old_values))
        return re.sub(pattern, new_value, string)


def _process_doc_fewshot(doc, num_fewshot):
    n = len(next(v for v in doc.values()))

    queries = []
    targets = []
    for i_doc in range(n):
        arange = set(range(n))
        examples = random.sample(tuple(arange - {i_doc}), num_fewshot)
        prompts = []
        for idx, i_example in enumerate(examples):
            prompts.append(
                FEWSHOT_PROMPT.format(
                    index=idx + 1,
                    instruction=doc["objektif"][i_example],
                    choices=doc["soalan"][i_example],
                    answer=doc["jawapan"][i_example],
                )
            )
        prompts.append(
            QUERY_PROMPT.format(
                instruction=doc["objektif"][i_doc],
                choices=doc["soalan"][i_doc],
            )
        )
        queries.append("\n\n".join(prompts))
        targets.append([doc["jawapan"][i_doc]])

    return {"query": queries, "target": targets}

