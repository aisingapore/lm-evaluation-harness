import functools

import datasets


PROMPT = [
    '请仔细阅读以下问题并从选项中选择最合适的答案，仅回答相应选项，不需要解释。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n',
    '请仔细阅读以下问题，并直接给出正确答案的选项，不需要解释。\n问题:\n{}\n选项:\n{}\n答案:\n',
    '针对以下问题选择正确答案，请直接选择正确的选项，不需要解释。\n问题:\n{}\n选项:\n{}\n答案:\n',
    '分析并从以下提供的选项中选择唯一的正确答案，如不确定，则选择你认为最可能的答案，不需要解释。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n',
    '请认真阅读以下中文多选题，并给出正确答案，不需要解释。\n\n问题:\n{}\n\n选项:\n{}\n\n答案:\n'
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

