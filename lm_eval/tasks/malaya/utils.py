import random

import datasets

def preprocess_text(doc):
     # "objektif: {{objektif}}\nsoalan: {{soalan}}?\njawapan:\n"
     return f"objektif: {doc['objektif']}\nsoalan: {doc['soalan']}?\njawapan:\n"

def doc_to_target(doc):
    return doc['jawapan']

def _process_wrapper(dataset: datasets.Dataset, num_fewshots:int):
    arange = set(range(len(dataset)))
    def _process_doc(doc:dict) -> dict:
        prompts = []
        # curr idx
        i = int(doc['no']) - 1
        # sample N other indices
        shots = random.sample(list(arange - {i}), num_fewshots)
        for no, s in enumerate(shots, start = 1):
            prompts.append(f'Contoh soalan {no}\n{preprocess_text(dataset[s])}{doc_to_target(dataset[s])}')
        
        prompts.append(preprocess_text(doc))
        prompt = '\n\n'.join(prompts)
        return {'prompt': prompt}
    
    return dataset.map(_process_doc)

def one_shot(dataset: datasets.Dataset):
    return _process_wrapper(dataset, 1)

def three_shot(dataset: datasets.Dataset):
    return _process_wrapper(dataset, 3)
    