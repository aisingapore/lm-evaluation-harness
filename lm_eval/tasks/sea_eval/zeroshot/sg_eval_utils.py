import functools

import datasets


PROMPT = [
    'Please carefully read the following question and select the most appropriate answer from the choices. Simply select the choice, no explanations required.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n',
    'Read the following question carefully and select the correct answer from the choices. Simply select the choice, no explanations required.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n',
    'Please select the most appropriate option to answer the question from your perspective as a resident of Singapore. Simply select the choice, no explanations required.\nQuestion:\n{}\nChoices:\n{}\nAnswer:\n',
    'Please answer the following Singapore-related questions by selecting the most probable answer from the choices. Simply select the choice, no explanations required.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n',
    'As a person living in Singapore, try your best to answer the question by selecting the most appropriate option. Simply select the choice, no explanations required.\n\nQuestion:\n{}\n\nChoices:\n{}\n\nAnswer:\n'
]


def _process_doc(prompt, doc):
    out_doc = {
        "query": prompt.format(doc["question"], "\n".join(doc["choices"])),
        "align_target": "{}\n{}".format("\n".join(doc["choices"]), doc["answer"]),
    }
    return out_doc


def process_docs_prompt_0(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[0])
    return dataset.map(process_doc_partial)


def process_docs_prompt_1(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[1])
    return dataset.map(process_doc_partial)


def process_docs_prompt_2(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[2])
    return dataset.map(process_doc_partial)


def process_docs_prompt_3(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[3])
    return dataset.map(process_doc_partial)


def process_docs_prompt_4(dataset: datasets.Dataset) -> datasets.Dataset:
    process_doc_partial = functools.partial(_process_doc, PROMPT[4])
    return dataset.map(process_doc_partial)

