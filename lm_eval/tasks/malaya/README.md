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

1. `do_sample` for generation must be set to false. 
2. Use first N shot for few shot learning, instead of default random.

```
# lm evaluation harness
diff --git a/lm_eval/tasks/malaya/_default.yaml b/lm_eval/tasks/malaya/_default.yaml
index e30588e2..ae586560 100644
--- a/lm_eval/tasks/malaya/_default.yaml
+++ b/lm_eval/tasks/malaya/_default.yaml
@@ -5,13 +5,13 @@ doc_to_text: !function utils.doc_to_text
 doc_to_target: jawapan
 output_type: generate_until
 process_results: !function utils.process_results
-repeats: 5
+repeats: 1
 generation_kwargs:
   max_new_tokens: 3
   top_p: 0.95
   top_k: 50
   temperature: 0.5
-  do_sample: true
+  do_sample: false
   num_beams: 1
   repetition_penalty: 1.05
   max_length: null
diff --git a/lm_eval/tasks/malaya/utils.py b/lm_eval/tasks/malaya/utils.py
index e215a898..5939fd23 100644
--- a/lm_eval/tasks/malaya/utils.py
+++ b/lm_eval/tasks/malaya/utils.py
@@ -23,9 +23,9 @@ def _process_wrapper(dataset: datasets.Dataset, num_fewshots: int):
         prompts = []
         # curr idx
         # sample N other indices
-        shots = random.sample(sorted(arange - {idx}), num_fewshots)
+        # shots = random.sample(sorted(arange - {idx}), num_fewshots)
         #TODO: uncomment the below line to use first N indices instead of random
-        # shots = sorted(arange - {idx})[:num_fewshots]
+        shots = sorted(arange - {idx})[:num_fewshots]
         for no, s in enumerate(shots, start=1):
             prompts.append(
                 f"Contoh soalan {no}\n{doc_to_text(dataset[s])} {doc_to_target(dataset[s])}"
```

For llm-benchmarks

```
diff --git a/evaluate.py b/evaluate.py
index a8be841..103f78d 100644
--- a/evaluate.py
+++ b/evaluate.py
@@ -99,7 +99,8 @@ def run_test(args, model, tokenizer, questions, n_shots):
         prompts = []
         if n_shots:
             arange = set(range(len(questions)))
-            shots = random.sample(arange - {i}, n_shots)
+            # shots = random.sample(sorted(arange - {i}), n_shots)
+            shots = sorted(arange-{i})[:n_shots]
             for no, s in enumerate(shots):
                 prompts.append(f'Contoh soalan {no + 1}\n' + convert_prompt(questions[s], answer = True))
         prompts.append(convert_prompt(questions[i]))
@@ -115,7 +116,7 @@ def run_test(args, model, tokenizer, questions, n_shots):
                     top_p=0.95,
                     top_k=50,
                     temperature=0.5,
-                    do_sample=True,
+                    do_sample=False,
                     num_beams=1,
                     repetition_penalty=1.05,
                 )
@@ -125,6 +126,8 @@ def run_test(args, model, tokenizer, questions, n_shots):
         
             except Exception as e:
                 print(e)
+                # necessary if all iterations fail, for most_common
+                repeat.append("")
                 pass
         
         questions[i]['output'] = repeat
```
