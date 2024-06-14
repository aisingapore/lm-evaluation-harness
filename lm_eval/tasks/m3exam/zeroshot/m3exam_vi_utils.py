import datasets

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
            [] if not doc["background"] else [doc["background"]]
        )
        doc["query"] = PROMPT.format(
            subject=SUBJECT_TO_TARGET[doc["subject_category"]],
            background="\n".join(doc["background_description"]),
            question=doc["question_text"],
            options="\n".join(doc["options"]),
        )
        doc["align_target"] = doc["answer_text"]
        return doc

    return dataset.map(_process_doc)

def mean(items):
    match = 0
    total = 0
    errors = []
    illformats = []

    for item in items:
        answer = item[0].strip()
        # Postprocess based on main.py
        res = item[1].split(ANS_KEYWORD)[-1].strip().split()[0]
        pred = res.strip()
        if len(pred) > 1:
            if pred[0] != "(":
                pred = pred[0]   # Assume answer is A) xxxx
            else:
                pred = pred[1]   # Assume answer is (A) xxxx
        if answer == pred:
            match += 1
        total += 1

    return match / total
