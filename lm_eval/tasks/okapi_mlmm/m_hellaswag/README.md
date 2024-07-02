# m\_hellaswag

### Paper

Title: `Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback`

Abstract: https://arxiv.org/pdf/2307.16039

Homepage: https://github.com/nlp-uoregon/mlmm-evaluation


### Citation

```
@article{dac2023okapi,
  title={Okapi: Instruction-tuned Large Language Models in Multiple Languages with Reinforcement Learning from Human Feedback},
  author={Dac Lai, Viet and Van Nguyen, Chien and Ngo, Nghia Trung and Nguyen, Thuat and Dernoncourt, Franck and Rossi, Ryan A and Nguyen, Thien Huu},
  journal={arXiv e-prints},
  pages={arXiv--2307},
  year={2023}
}
```

### Groups and Tasks

#### Groups

* `hellaswag_mmlu`: Multi lingual HellaSwag. Supports the Indonesian `id`, Tamil `ta`, Vietnamese `vi`, and Chinese `zh`.

#### Tasks

* `hellaswag_mlmm_{lang}`: HellaSwag task in the specified language

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?
