import functools
import itertools
import random

import datasets

LANGS = ['Chinese', 'Indonesian', 'Spanish', 'Vietnamese', 'Malay', 'English', 'Filipino']
FEWSHOT_PROMPT = "Question:\n{}\n\nChoices:\n{}\n\nAnswer:\n{}\n\n"
QUERY_PROMPT = "Question:\n{}\n\nChoices:\n{}\n\nAnswer:\n"


def _process_doc(doc):
    n = len(next(v for v in doc.values()))

    raw_data = []
    for i in range(n):
        raw_data.append({key: doc[key][i] for key in doc})

    all_samples = []
    for sample_set in raw_data:
        for key in sample_set:
            if key == 'id':
                continue
            all_samples.append(sample_set[key])

    queries = []
    targets = []
    random.seed(1234)
    for sample_set in raw_data:
        for key in sample_set:
            if key == 'id':
                continue

            five_plus_one_samples = random.sample(all_samples, 6)

            count = 0
            query = ""
            for sample in five_plus_one_samples:
                # Filter out the sample with the same context
                if sample['question'] != sample_set[key]['question']:
                    query += FEWSHOT_PROMPT.format(
                        sample["question"], "\n".join(sample["choices"]), sample["answer"]
                    )
                    count += 1
                if count == 5:
                    break

            query += QUERY_PROMPT.format(
                sample_set[key]['question'], "\n".join(sample_set[key]['choices'])
            )
            queries.append(query)
            targets.append(
                "{}\n{}\n{}\n{}".format(
                    sample_set["id"],
                    key,
                    "\n".join(sample_set[key]["choices"]),
                    sample_set[key]["answer"],
                )
            )

    return {"query": queries, "align_target": targets}


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    return dataset.map(
        _process_doc, batched=True, remove_columns=dataset.column_names
    )
