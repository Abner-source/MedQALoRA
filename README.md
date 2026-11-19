# MedQALoRA: Does cleaner data make a better medical Q&A model?

A small controlled experiment. I fine-tuned the same small open-source LLM (Qwen2.5-1.5B-Instruct) with LoRA twice on medical question-answer data: once on a **raw** sample and once on a **curated** sample of the same size. Then I compared both against the untouched **base** model on held-out questions.

**Short answer:** curation changed *how* the model answers (shorter, more focused, stops on time, closer to expert wording), but neither fine-tune made the answers more *factually correct*.

| | base | raw | curated |
|---|---:|---:|---:|
| Judge correctness (1–5) | **3.18** | 2.82 | 2.79 |
| Answers that never stopped | 32.7% | 28.7% | **22.0%** |
| ROUGE-L vs expert answer | 0.17 | 0.28 | **0.31** |
| Median answer length (words) | 166.5 | 131 | 116.5 |

---

## Setup

| | |
|---|---|
| Dataset | [MedQuAD](https://github.com/abachaa/MedQuAD): 16,407 medical Q&A pairs from NIH sources |
| Base model | [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Method | LoRA fine-tuning (Hugging Face) |
| Training size | 2,000 examples per run (raw and curated) |
| Compute | Google Colab, free T4 GPU |
| Test set | 150 held-out questions, removed from training data along with near-copies |

## 1. Data curation

Each filter was applied in order. The table shows how many rows were left after each one.

| Step | Rows left |
|---|---:|
| Loaded | 16,407 |
| Answer ≥ 30 words | 15,354 |
| Answer ≤ 300 words | 12,616 |
| Mostly prose (not lists/tables) | 12,613 |
| Exact duplicate questions removed | 11,219 |
| Exact duplicate answers removed | 10,767 |
| Near-duplicate answers removed (Jaccard ≥ 0.8) | 10,142 |
| Test set + near-copies of test removed | 9,992 |
| Balanced sample (no question type > 20%) | **2,000** |

### Raw vs curated training sets

| | raw | curated |
|---|---:|---:|
| Rows | 2,000 | 2,000 |
| Median answer words | 140 | 110 |
| % answers > 300 words | 17.3 | 0 |
| % answers < 30 words | 6.8 | 0 |
| % duplicate questions | 2.4 | 0 |
| Largest question type % | 26.4 | 7.8 |

### Truncation

Examples longer than the max sequence length get cut off during training, and the cut-off examples lose their end-of-sequence (stop) token. The model never sees those answers end.

| | Truncated examples |
|---|---:|
| raw | 300 / 2,000 (**15.0%**) |
| curated | 4 / 2,000 (**0.2%**) |

## 2. Training

![Loss curves](results/loss_curves.png)

Both runs trained for the same number of steps. The raw model reaches a lower validation loss on the practice set. Lower loss did **not** translate into better judged answers (see below), so I treat loss as a training-health check, not a quality measure.

## 3. Evaluation

Each model answered the same 150 held-out questions. Metrics:

- **Correctness and completeness (1–5):** scored by an LLM judge (`gpt-4.1-mini`) against the expert reference answer
- **Safe %:** share of answers the judge marked as safe
- **ROUGE-L:** word-sequence overlap with the expert answer
- **Median words:** typical answer length
- **Rambling %:** answers that hit the token limit without producing a stop token

**Judge check:** I hand-graded 20 random answers blind, against the same reference answers and a 1–5 rubric. I gave the same score as the judge on **13/20** and was within one point on **20/20**; on average the judge scored only 0.15 points higher than me. The correctness scores can be treated as reliable.

### Results

| model | correctness | completeness | safe % | ROUGE-L | median words | rambling % |
|---|---:|---:|---:|---:|---:|---:|
| base | 3.18 | 2.61 | 98.67 | 0.17 | 166.5 | 32.67 |
| raw | 2.82 | 2.45 | 100 | 0.28 | 131 | 28.67 |
| curated | 2.79 | 2.43 | 100 | 0.31 | 116.5 | 22.0 |

Bootstrap 95% intervals for correctness:

| model | correctness | 95% interval |
|---|---:|---|
| base | 3.18 | [2.96, 3.40] |
| raw | 2.82 | [2.59, 3.03] |
| curated | 2.79 | [2.57, 2.99] |

The raw and curated intervals almost fully overlap, so curation made **no measurable difference to correctness**. Both fine-tunes score below base, and their intervals only just overlap with base's.

![Results](results/results_chart.png)

### Example

**Q: What is (are) Ohtahara Syndrome?**

> **Expert reference:** A neurological disorder of newborns, usually within the first three months of life, characterised by epileptic (mainly tonic) seizures…

| model | correctness | stopped? | answer (summary) |
|---|:---:|:---:|---|
| base | 1 | yes | Describes a diarrhoea and dehydration condition in children under 5. Wrong. |
| raw | 2 | yes | Correctly calls it a nervous-system disorder with seizures, but gives the wrong onset age (6 months–3 years) and an invented gene cause. |
| curated | 1 | **no** | Fluent and expert-sounding, but wrong onset age and symptoms, and runs until the token limit. |

All three answers sound confident. None is reliable. This is the core finding in one example.

## What I learned

1. **Fine-tuning teaches format, not facts.** Length, focus, wording and stopping behaviour all moved toward the expert answers. Factual correctness did not improve.
2. **Data curation has measurable effects.** Removing very long answers cut truncation from 15% to 0.2%, and the curated model rambled less (22% vs 29%) with shorter, closer-to-reference answers.
3. **Lower loss ≠ better answers.** The raw run had lower validation loss, but the judged quality was the same.
4. **Fluent answers fool people too.** On my first hand-grading pass I scored several confident, well-written answers highly even though they named the wrong gene or the wrong type of disorder. Grading fact by fact against the reference fixed this. It's the same trap the fine-tuned models fall into.
5. **Small models need grounding for medical facts.** A fluent, wrong medical answer is worse than a short, uncertain one. Retrieval (RAG) over trusted sources is the natural next step.

## Limitations

- One base model, one seed and one run per condition
- 150 test questions; the intervals are wide
- Correctness comes from an LLM judge; it agreed closely with my grades, but on a sample of only 20 answers
- Not a medical tool. Do not use these outputs for medical decisions.

## Next steps

- Compare against a RAG baseline over the same MedQuAD sources
- Repeat with multiple seeds to tighten the intervals
- Try a larger base model to see whether the format-vs-facts gap holds

## How to run

1. Open the notebook in Google Colab with a T4 GPU runtime.
2. Mount Google Drive. Results are saved to `results/`.
3. Run Parts 1–9 in order: data loading, curation, tokenisation and truncation check, LoRA training (raw and curated), evaluation, and bootstrap intervals.
