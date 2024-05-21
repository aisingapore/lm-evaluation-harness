import functools
import itertools
import random

import datasets


LANGS = ['Chinese', 'Indonesian', 'Spanish', 'Vietnamese', 'Malay', 'English', 'Filipino']
FEWSHOT_PROMPT = "Question:\n{}\n\nChoices:\n{}\n\nAnswer:\n{}\n\n"
QUERY_PROMPT = "Question:\n{}\n\nChoices:\n{}\n\nAnswer:\n"

def _process_doc(doc):
    # Merges a list of list in a zig-zag fashion, e.g.,
    # [[a1, a2], [b1, b2], [c1, c2]] -> [a1, b1, c1, a2, b2, c2]
    def _zig_zag_merge(arr):
        return list(itertools.chain(*list(zip(*arr))))

    ids = []
    queries = []
    targets = []
    n = len(doc["id"])
    past_indices = []
    random.seed(1234)
    for lang in LANGS:
        lang_ids = []
        lang_queries = []
        lang_targets = []
        for i, sample in enumerate(doc[lang]):
            if i == len(past_indices):
                indices = random.sample(range(n), 6)
                past_indices.append(indices)
            else:
                indices = past_indices[i]
            lang_ids.append(f"{lang}_{doc['id'][i]}")
            count = 0
            query = ""
            for idx in indices:
                if doc[lang][idx]["question"] != sample["question"]:
                    query += FEWSHOT_PROMPT.format(
                        doc[lang][idx]["question"],
                        "\n".join(doc[lang][idx]["choices"]),
                        doc[lang][idx]["answer"],
                    )
                    count += 1
                if count == 5:
                    break;
            query += QUERY_PROMPT.format(sample["question"], "\n".join(sample["choices"]))
            lang_queries.append(query)
            #  lang_queries.append(
            #      prompt.format(sample["question"], "\n".join(sample["choices"]))
            #  )
            lang_targets.append(
                "{}\n{}\n{}\n{}".format(
                    doc['id'][i], lang, "\n".join(sample["choices"]), sample["answer"]
                )
            )
        ids.append(lang_ids)
        queries.append(lang_queries)
        targets.append(lang_targets)

    ids = _zig_zag_merge(ids)
    queries = _zig_zag_merge(queries)
    targets = _zig_zag_merge(targets)

    return {"new_id": ids, "query": queries, "align_target": targets}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(
        _process_doc, batched=True, remove_columns=dataset.column_names
    )
