import functools

import datasets


PROMPT = [
    'Translate the following sentence from Indonesian to English. Output the translation only, nothing else.\n\nSentence in Indonesian:\n{}\n\nTranslation in English:\n',
    'Please translate the provided Indonesian text into English. Output the translated content only.\nSentence in Indonesian:\n{}\nTranslation in English:\n',
    'Translate the Indonesian text provided into English and provide only the translated content.\nSentence in Indonesian:\n{}\nTranslation in English:\n',
    'Given the sentence below, perform machine translation from Indonesian to English. Output the translated content only.\n\nSentence in Indonesian:\n{}\n\nTranslation in English:\n',
    'Please translate the sentence: \"{}\" from Indonesian to English. Output the translated content only.\n\nTranslation in English:\n'
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

