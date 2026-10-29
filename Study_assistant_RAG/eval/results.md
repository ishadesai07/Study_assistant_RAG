# Evaluation results

28 questions over `attention_is_all_you_need.pdf` (15 pages).

| config | chunks | page@1 | page@2 | page@4 | page@8 | evid@1 | evid@2 | evid@4 | evid@8 | mrr |
|---|---|---|---|---|---|---|---|---|---|---|
| chunk=128/16, bge-small | 110 | 0.89 | 0.96 | 1.00 | 1.00 | 0.46 | 0.64 | 0.71 | 0.79 | 0.58 |
| chunk=256/32, bge-small | 56 | 0.82 | 0.96 | 1.00 | 1.00 | 0.46 | 0.71 | 0.75 | 0.79 | 0.61 |
| chunk=512/64, bge-small | 29 | 0.64 | 0.96 | 0.96 | 1.00 | 0.57 | 0.75 | 0.79 | 0.86 | 0.68 |
| chunk=1024/128, bge-small | 15 | 0.64 | 0.75 | 0.82 | 1.00 | 0.54 | 0.64 | 0.71 | 0.86 | 0.63 |
| chunk=512/64, MiniLM-L6 | 29 | 0.61 | 0.82 | 0.93 | 0.96 | 0.39 | 0.61 | 0.71 | 0.75 | 0.54 |

Best chunking by evid@4 then MRR: **chunk=512/64, bge-small**.
