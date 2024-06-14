# Task-name

### Paper

Title: `M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models`

Abstract: https://arxiv.org/abs/2306.05179

We introduce M3Exam, a novel benchmark sourced from real and official human exam questions for evaluating LLMs in a multilingual, multimodal, and multilevel context.

Homepage: https://github.com/DAMO-NLP-SG/M3Exam


### Citation

```
@article{zhang2023m3exam,
      title={M3Exam: A Multilingual, Multimodal, Multilevel Benchmark for Examining Large Language Models},
      author={Wenxuan Zhang and Sharifah Mahani Aljunied and Chang Gao and Yew Ken Chia and Lidong Bing},
      year={2023},
      eprint={2306.05179},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Groups and Tasks

#### Groups

* `m3exam_zeroshot`

#### Tasks

* `m3exam_zeroshot_vi`: Vietnamese subset with Vietnamese prompt, corresponds to the "Monolingual" strategy from Table 3.

### Reproducibility

To reproduce the original implementation's results

* Set `batch_size=1`
* Ensure `max_new_token=3` and `until=[]` in `generation_kwargs`

### Checklist

For adding novel benchmarks/datasets to the library:
* [x] Is the task an existing benchmark in the literature?
  * [x] Have you referenced the original paper that introduced the task?
  * [x] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
