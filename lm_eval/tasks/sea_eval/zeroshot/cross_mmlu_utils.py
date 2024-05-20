import functools
import itertools

import datasets


LANGS = ['Chinese', 'Indonesian', 'Spanish', 'Vietnamese', 'Malay', 'English', 'Filipino']
PROMPT = [
    'Respond to the question by selecting the most appropriate answer. Simply select the answer, no explanations required.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n',
    'Kindly choose the correct answer from the options provided for the multiple-choice question. Simply select the answer, no explanations required.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n',
    'Solve the multi-choice question by selecting the accurate answer. Simply select the answer, no explanations required.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n',
    'Please answer the following multiple-choice question by selecting the correct option. Simply select the answer, no explanations required.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n',
    'As an expert, your task is to solve the following multiple-choice question. Identify the correct response among the given choices. Simply select the answer, no explanations required.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n'
]

def _process_doc(prompt, doc):
    # Merges a list of list in a zig-zag fashion, e.g.,
    # [[a1, a2], [b1, b2], [c1, c2]] -> [a1, b1, c1, a2, b2, c2]
    def _zig_zag_merge(arr):
        return list(itertools.chain(*list(zip(*arr))))

    ids = []
    queries = []
    targets = []
    for lang in LANGS:
        lang_ids = []
        lang_queries = []
        lang_targets = []
        for i, sample in enumerate(doc[lang]):
            lang_ids.append(f"{lang}_{doc['id'][i]}")
            lang_queries.append(
                prompt.format(sample["question"], "\n".join(sample["choices"]))
            )
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


def process_docs_prompt_0(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[0])
    return dataset.map(
        process_doc_partial, batched=True, remove_columns=dataset.column_names
    )


def process_docs_prompt_1(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[1])
    return dataset.map(
        process_doc_partial, batched=True, remove_columns=dataset.column_names
    )


def process_docs_prompt_2(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[2])
    return dataset.map(
        process_doc_partial, batched=True, remove_columns=dataset.column_names
    )


def process_docs_prompt_3(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[3])
    return dataset.map(
        process_doc_partial, batched=True, remove_columns=dataset.column_names
    )

def process_docs_prompt_4(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[4])
    return dataset.map(
        process_doc_partial, batched=True, remove_columns=dataset.column_names
    )
