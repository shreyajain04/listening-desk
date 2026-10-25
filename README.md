# Listening Desk (Team C2, Project 3)

Social listening and crisis response for IndiGo mentions. **Unofficial educational prototype**, not affiliated with IndiGo. All posts and numbers are synthetic. Not legal, PR or financial advice.

To put it online, follow **DEPLOY.md** (about an hour, nothing to install).

## What it does

1. **Input guard** hides phone numbers, emails and booking codes, caps length and flags manipulation attempts.
2. **LLM classifier** labels sentiment, topic, urgency, sarcasm and language. Groq first, Gemini as free fallback, keyword rules if neither answers.
3. **Spread model** gives the chance of 200+ reshares in 24 hours, from data known 60 minutes after posting.
4. **Router** sends negative posts above the alert threshold, and every safety post, to crisis alerts. Manipulation attempts go to a reviewer.
5. **Drafter** writes a reply from retrieved playbook sections, then checks it for promises, personal data, citations and length.
6. **A person** acknowledges crisis alerts and approves or rejects every reply. The app cannot post.

## Folder map

| Path | What it is | Owner |
|---|---|---|
| `app.py` | All screens | Ananya |
| `core/llm.py` | Groq and Gemini calls, behind one function | Aatif |
| `core/classifier.py` | Classifier prompt, validation, offline rules | Aatif |
| `core/retrieval.py` | Finds playbook sections | Aatif |
| `core/drafter.py` | Writes reply drafts | Aatif |
| `core/guards.py` | Redaction and draft checks | Shreya |
| `core/router.py` | Business rules for crisis, reply, routine | Abhimanyu |
| `core/spread.py` | Loads and explains the spread model | Abhimanyu |
| `data/playbook/*.md` | Tone guide and crisis playbook | Vaishnavi |
| `data/mentions_demo.csv` | The 25 inbox posts | Vaishnavi |
| `training/` | Synthetic data generator and model training | Abhimanyu |
| `models/`, `eval/` | Saved model, metrics, test-set scores | Abhimanyu |

## How the planned upgrades slot in

Each upgrade changes one file and keeps its function names, so the rest of the app is untouched.

| Upgrade | Where |
|---|---|
| Better playbook text | Edit `data/playbook/*.md`. Keep the `## [ID] Title` headings and `Template:` lines. |
| Embedding search with fastembed | Replace the inside of `retrieve()` in `core/retrieval.py` |
| More demo posts | Add rows to `data/mentions_demo.csv` (same columns) |
| Classifier evaluation on the gold set | New script in `eval/`; results shown on the Model results tab, which already has a placeholder |
| Red-team results | New script in `eval/`; placeholder already on the Model results tab |
| Model v2 (15 and 30 minute scoring) | `training/` scripts, then replace the files in `models/` |
| A different LLM | Change `GROQ_MODEL` or `GEMINI_MODEL` in Streamlit secrets |

## Retrain the spread model

```
python training/generate_synthetic_mentions.py
python training/train_spread_model.py
```

Same seed and versions give the same numbers as the progress report.

## Run on your own laptop (optional)

```
pip install -r requirements.txt
streamlit run app.py
```

Without keys, the app runs on keyword rules and playbook templates. To use the LLM locally, put your keys in `.streamlit/secrets.toml`, which `.gitignore` keeps out of GitHub.

## Two numbers worth knowing for the demo

- The model was trained with balanced class weights, so its raw scores are inflated. The app corrects them back to real chances using the 4.8% base rate. On the test set the corrected scores average 4.6%.
- The alert threshold is about 1.9%. That matches the cost logic: alert whenever the chance exceeds 500 ÷ (500 + 25,000) ≈ 2%, because a missed crisis is assumed to cost 50 times a false alarm.
