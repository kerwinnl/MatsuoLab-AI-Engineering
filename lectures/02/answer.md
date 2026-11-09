# Lecture 02 Answers

## Exercises

- Model and settings used (from `config` in `result.json`):
- Did you run training yourself (`executed_training_myself`): Yes / No (how far you ran the notebook: )
- Before training: `format_rate` = / `content_rate` =
- After training: `format_rate` = / `content_rate` =
- Questions that regressed:
- Failures remaining after training and their types (assertion without evidence / evasion through a confirmation request despite evidence being present / broken format):
- Results of knowledge questions (section 6; classification before → after training) and observations:
- What changed and what stayed the same with training:

## Homework

## Homework

- Level attempted: **Advanced**
- Settings or data changed:
  - Training data: ran `uv run python scripts/make_sft_dataset.py > glossary_sft.jsonl` (65 terms × 2 templates = 130 plain-text Q&A pairs). The script writes bare definitions, so I wrapped each as `{"answer": <definition>, "needs_review": false}` to keep the exercise's JSON format and `needs_review` design.
  - Added 24 **ungrounded** training examples (12 terms not in the glossary × 2 templates, `needs_review: true`). Held out 6 "Briefly explain X." questions + 2 ungrounded as validation. Train = 148, valid = 8.
  - Prompt is **closed-book**: only the question is given, no reference text (the FAQ exercise put the FAQ in the prompt).
  - Settings: `num_train_epochs` 20 → 8, `warmup_steps` 6 → 10, `max_new_tokens` 96 → 160 (definitions are up to 59 words; 96 tokens would cut off the JSON). Other CONFIG unchanged (lr 2e-4, r 16, alpha 32, seed 42, greedy decoding).
  - Rebuilt test and gold: 42 questions, one required phrase per grounded question, copied from the definition (asserted in code).

| Kind | n | Example | Purpose |
|---|---|---|---|
| direct | 12 | "What is LoRA?" | same wording as training |
| rephrase | 12 | "I keep hearing about LoRA. What does it mean?" | new wording (templates unused in training); term still named |
| alias | 6 | "What is HITL?" | glossary aliases, never in training |
| descriptive | 4 | "Which method freezes pretrained weights and trains only small low-rank matrices?" | term never named |
| ungrounded | 8 | "What is KV cache?", "What is RLHF?" | not in glossary (checked against all terms and aliases); disjoint from training/validation ungrounded terms |

- Results (content_rate by kind; same 42 questions, greedy decoding):

| System | direct | rephrase | alias | descriptive | ungrounded | ALL content | format_rate | ungrounded asserted (of 8) |
|---|---|---|---|---|---|---|---|---|
| Base model (before) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 | 6 |
| LoRA on glossary (after) | 1.00 | 1.00 | 0.17 | 0.00 | 0.38 | 0.67 | 1.00 | 5 |
| Glossary search (no training) | 1.00 | 1.00 | 1.00 | 1.00 | 0.88 | 0.98 | 1.00 | 1 |

  Training: 6.39M trainable parameters (0.84% of 758.8M), 152 steps (8 epochs), 613 s on a T4. Training loss 2.30 (step 10) → 0.98 (20) → 0.21 (60) → 0.05 (100) → 0.0405 (step 150, last logged). Regressions: none (and none were possible, since nothing passed content before training).

  **Observations**
  1. **Format was not the problem.** format_rate was already 1.0 before training (closed-book with the JSON instruction). The base model scored 0.00 on content because it does not know the glossary; training changed *knowledge of the glossary*, not the format.
  2. **Memorisation worked for named terms.** direct and rephrase both went 0.00 → 1.00. Rewording did not hurt as long as the term name appears in the question.
  3. **It did not generalise past the exact term.** alias 0.17 (1/6 passed; "lr" by elimination). "Low-Rank Adaptation" was wrongly refused (needs_review true). The other four got invented definitions, e.g. HITL described as a terminal app installed with `pip install hitl`, and "429 error" as a runner-permission error. descriptive was 0/4 under the strict phrase check: three answers were wrong (e.g. "Iterative training", "LoRA align"), and the Overfitting answer was conceptually correct but missed the required phrase, so a looser scorer would give about 1/4.
  4. **Unknown terms were mostly asserted.** 5 of 8 ungrounded questions got `needs_review: false` (KV cache, DoRA, RLHF, Few-shot prompting, Weight decay); only 3 were refused (Retrieval-augmented generation, Chain of thought, Prompt caching, by elimination). Asserted count barely moved from before (6) to after (5), but the wrong answers are now written in the glossary's style and mix in real course vocabulary and fake specifics (e.g. DoRA "released in October 2026", KV cache as a Pydantic AI "memory" tool), which makes them more convincing.
  5. **Search baseline** (TF-IDF over term + aliases + definition + exact-title boost; threshold 5.27 from validation only): 0.98. Its one failure is "What is Prompt caching?", matched to the entry "Prompt" (a confident wrong nearest-neighbour). Caveats: search's index contains the aliases, while the script's training data did not (adding alias questions to the training data would be the fair next experiment; I did not run it), and most test questions name the term.

  **Section 6 knowledge questions** (absent from glossary and training; desired = needs_review):

| Question | Before | After (output) |
|---|---|---|
| What is the instructor's name? | asserted ("Dr. Sarah Jenkins") | asserted ("Alexio Maurensi") |
| What is the topic of the next lecture? | asserted (AI ethics and governance) | asserted (a made-up topic about evaluation methods and detecting AI-generated content) |
| When is the final assignment deadline? | asserted ("10:00 PM on Friday, October 25, 2026") | asserted ("31 December 2026. Late submissions will not be graded.") |
| What is LoRA? (control, in glossary) | asserted (generic textbook answer, mentions BERT) | asserted (the exact glossary definition) |

  All three unknown questions were asserted both before and after training (3/3 → 3/3, none changed to needs_review), and the invented answers are specific and confident (a name, a date, a lecture topic). For the control, an assertion is the desired behaviour; the difference is that the base model gave a generic answer while the tuned model returned the glossary text. Report for this data: training taught the glossary content but did not teach the model to refuse questions outside it, even though it had seen 12 ungrounded terms with needs_review=true.

- Discussion of when fine-tuning is and is not suitable:
  - **Glossary content is knowledge, not behaviour.** Training stored the named definitions well (100% on direct/rephrase), but that is recall of the exact training text. Anything slightly off the trained path (aliases, descriptions) produced fluent fabrications. Updating one definition means rebuilding data, retraining and re-evaluating; in search it is one edit, effective immediately, with a citable source.
  - **Training gave no reliable "I don't know".** With 12 ungrounded training terms the model refused 3/8 unknown terms and asserted 5/8. Search gets a refusal from a simple rule (no match above a threshold), though nearest-term confusion (Prompt vs Prompt caching) still happens. Both need ungrounded questions in the evaluation set.
  - **Where fine-tuning does fit:** stable output format, tone or procedure; shorter prompts; a small or local model at high volume. For this glossary, search should hold the facts; fine-tuning would at most shape the output style on top.
  - **Checks I made:** only the `direct` test kind reuses training wording; ungrounded terms are disjoint across train/valid/test; same prompt, decoding and scoring before and after; scoring is strict (required phrase), so paraphrased but correct answers count as failures.
