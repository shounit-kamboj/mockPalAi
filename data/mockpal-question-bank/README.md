# MockPal question bank

4,055 unique interview questions for MockPal AI, one question per record, collected from
23 openly licensed GitHub repositories (see `ATTRIBUTION.md`).

## Files

- `questions.json`: the bank as a JSON array (use this for ingestion)
- `questions.csv`: the same data for spreadsheets; `skills` is `;`-separated
- `ATTRIBUTION.md`: every source, its licence and how many questions it contributed
- `LICENSES/`: full licence text for each source
- `tools/`: the scripts that built this file (extract, tag and de-duplicate), if you want to rebuild or extend it

## Fields

| Field | Meaning |
|---|---|
| `id` | UUIDv5 of the normalized question text, so re-ingesting the same question updates instead of duplicating |
| `question` | The question, as written in the source (formatting removed) |
| `category` | `behavioural`, `technical`, `coding` or `system_design` |
| `topic` | Broad area: `backend`, `frontend`, `devops`, `machine_learning`, `behavioural`, `databases`, `cs_fundamentals`, `system_design`, `process`, `data_science` |
| `skills` | Tags from the source file plus keyword matching (e.g. `react`, `sql`, `teamwork`) |
| `difficulty` | `easy` / `medium` / `hard`, only where the source states it, otherwise null |
| `seniority` | `junior` / `mid` / `senior`, only where the source states it, otherwise null |
| `source`, `source_url`, `license` | Where the question came from and under which licence |

## Counts

- By category: technical 3,628, behavioural 243, coding 116, system_design 68
- By topic: backend 1,393, frontend 1,150, devops 476, machine_learning 296, behavioural 243, databases 194, cs_fundamentals 186, system_design 68, process 26, data_science 23
- By licence: MIT 3,487, CC-BY-4.0 199, GPL-2.0 169, Apache-2.0 95, GPL-3.0 79, BSD-3-Clause 26
- With a stated difficulty: 256; with a stated seniority: 440

## How it was built

1. Searched GitHub for interview-question collections and kept only repositories whose licence allows copying
   (MIT, BSD, Apache 2.0, CC BY 4.0, WTFPL, GPL). Repositories with no licence, or with non-commercial or
   no-derivatives terms, were skipped.
2. Pulled questions out of headings, lists, tables and summary blocks, English only. Dropped questions that
   only make sense next to a code snippet ("What is the output of the following code?") and "questions to ask
   the interviewer" sections.
3. Tagged category, topic and skills from the source file and keywords. Difficulty and seniority are only
   filled in when the source states them.
4. Removed exact and near-duplicate questions across sources (TF-IDF cosine similarity; stricter for short
   questions). One source (`aswanth6000/nodejs-interview-questions`) was mostly machine-written rewordings of
   the same topics, so it was collapsed from about 2,400 entries to its ~400 distinct questions.
