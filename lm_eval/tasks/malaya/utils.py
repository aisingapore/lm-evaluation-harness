import random
import datasets
import json
from typing import List


def doc_to_text(doc):
    return f"objektif: {doc['objektif']}\nsoalan: {doc['soalan']}\njawapan:"


def doc_to_target(doc):
    return doc["jawapan"]


def most_common(l: List) -> str:
    return max(set(l), key=l.count)


def _process_wrapper(dataset: datasets.Dataset, num_fewshots: int):
    arange = set(range(len(dataset)))

    def _process_doc(doc: dict, idx: int) -> dict:
        prompts = []
        # curr idx
        # sample N other indices
        shots = random.sample(sorted(arange - {idx}), num_fewshots)
        #TODO: uncomment the below line to use first N indices instead of random
        # shots = sorted(arange - {idx})[:num_fewshots]
        for no, s in enumerate(shots, start=1):
            prompts.append(
                f"Contoh soalan {no}\n{doc_to_text(dataset[s])} {doc_to_target(dataset[s])}"
            )

        prompts.append(doc_to_text(doc))
        prompt = "\n\n".join(prompts)
        return {"prompt": prompt}

    return dataset.map(_process_doc, with_indices = True)


def process_single_answer(r: str) -> str:
    r = r.strip().split()  # type: ignore
    if not r:
        return ""
    return r[0].replace(".", "").replace("</s>", "").split("\\")[0].split("/")[0]


def process_results(doc: dict, results: List[List[str]]):
    to_save = doc.copy()

    r = [process_single_answer(x) for x in results[0]]

    ans = most_common(r)
    gold = doc_to_target(doc)

    result_dict = {
        **({"acc": gold == ans}),
    }
    to_save["lm_output"] = results[0]
    to_save["lm_clean"] = r
    to_save["lm_answer"] = ans


    return result_dict


def one_shot(dataset: datasets.Dataset):
    return _process_wrapper(dataset, 1)


def three_shot(dataset: datasets.Dataset):
    return _process_wrapper(dataset, 3)
