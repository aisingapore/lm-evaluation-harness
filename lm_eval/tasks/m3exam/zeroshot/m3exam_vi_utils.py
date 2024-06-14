import datasets
from typing import Iterable, List, Union

from lm_eval.api.filter import Filter

SUBJECT_TO_TARGET = {
    "language": "Tiếng Việt",
    "math": "Toán",
    "social-science": "Khoa học xã hội",
    "natural-science": "Khoa học tự nhiên",
}
ANS_KEYWORD = "Câu trả lời:"
PROMPT = (
    "Sau đây là các câu hỏi trắc nghiệm về {subject}. Vui lòng chỉ đưa "
    "ra phương án đúng, không có bất kỳ chi tiết hay giải thích nào khác.\n\n"
    "{background}\n{question}\n{options}\nCâu trả lời:"
)


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        doc["background_description"] = (
            [] if not doc["background"] else ["", doc["background"]]
        )
        doc["query"] = PROMPT.format(
            subject=SUBJECT_TO_TARGET[doc["subject_category"]],
            background="\n".join(doc["background_description"]),
            question=doc["question_text"],
            options="\n".join(doc["options"]),
        )
        # Follow postprocessing in original eval.py implementation
        doc["align_target"] = doc["answer_text"].strip()
        return doc

    return dataset.map(_process_doc)


def mean(items):
    return sum(ans == pred for ans, pred in items) / len(items)


class MultiChoiceFilter(Filter):
    def apply(self, resps: Union[List, Iterable], docs: List[dict]) -> Iterable:
        def extract_choice(resp):
            """Attempts to extract multi-choice letter from the model's
            response.
            """
            # Model response postprocessing from original main.py implementation
            res = resp[0].split(ANS_KEYWORD)[-1].strip().split()[0]
            # Response alignments from original eval.py implementation
            res = res.strip()
            if len(res) > 1:
                if res[0] != "(":
                    res = res[0]  # Assumes A) xxxx
                else:
                    res = res[1]  # Assumes (A) xxxx
            return res

        return map(extract_choice, resps)
