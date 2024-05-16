import functools

import datasets


PROMPT = [
    'Summarize the following dialogue. Output the summary directly, nothing else.\n\nDialogue:\n{}\n\nSummary:\n',
    'Compose a concise summary by condensing the key points from the following dialogue. Output the summary directly, nothing else.\nDialogue:\n{}\nSummary:\n',
    'Please sum up the following conversation in a few sentences. Output the summary directly, nothing else.\nDialogue:\n{}\nSummary:\n',
    'Produce a brief summary of the following conversation, focusing on conveying essential information. Output the summary directly, nothing else.\n\nDialogue:\n{}\n\nSummary:\n',
    'Offer an extremely condensed summary of the conversation presented. Output the summary directly, nothing else.\n\nDialogue:\n{}\n\nSummary:\n'
    ]


def _process_doc(prompt, doc):
    out_doc = {
        "query": prompt.format(doc["context"]),
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

