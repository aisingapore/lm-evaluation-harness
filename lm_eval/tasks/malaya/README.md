# Malaya benchmarks (For internal use)

### Descriptions

Title: Mesolitica benchmarks

Tasks for mesolitica llm benchmarks.

Reference code: https://github.com/aisingapore/llm-benchmarks/tree/dev  
https://github.com/mesolitica/llm-benchmarks


### Citation

```
https://github.com/mesolitica/llm-benchmarks
```

### Deviations

To replicate the results from the original source code, here are some steps necessary:

1. Use `transformers.set_seed(1234)` in mesolitica and `--seed 1234` argument for lm-eval (not available in this branch)
2. lm-eval resets the seed on every task, so to replicate on mesolitica, every task needs to call `transformers.set_seed` again. 
3. `do_sample` for generation must be set to true. 
4. It may be necessary to use first_n samples for zero-shot, however. 
5. The results have been verified to be the same when done for 0 shot. 