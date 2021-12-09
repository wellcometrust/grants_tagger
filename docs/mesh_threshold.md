# Optimise threshold for MeSH

According to the paper "Threshold optimisation for multi-label classifiers"
by Pillai https://doi.org/10.1016/j.patcog.2013.01.012 an individual optimal
threshold exists per label in multilabel classification.

Here we present results with a constant vs individual optimised threshold
per model. As we have not managed to replicate the performance boost
from the literature we have not added this in `dvc.yaml`.

## Mesh XLinear

```
> grants_tagger evaluate model mesh-xlinear models/xlinear/model/ data/processed/test_mesh2021.jsonl models/xlinear/label_binarizer.pkl --threshold 0.1,0.2,0.3,0.4,0.5 --no-split-data

Threshold      P       R       F1
------------   -----   -----   -----

0.10           0.57    0.56    0.56
0.20           0.63    0.51    0.56
0.30           0.67    0.47    0.56
0.40           0.71    0.44    0.54
0.50           0.74    0.41    0.53

> grants_tagger tune threshold mesh-xlinear data/processed/test_mesh2021.jsonl models/xlinear/model/ models/xlinear/label_binarizer
.pkl models/thresholds.pkl --nb-thresholds 10

---Starting f1---
0.565

---Min f1---
0.493

Iteration 0 - f1 0.587 - time spent 21.54123330116272s
Iteration 1 - f1 0.590 - time spent 21.593909978866577s
Iteration 2 - f1 0.590 - time spent 21.93311047554016s
Iteration 3 - f1 0.590 - time spent 22.354877471923828s
---Optimal f1 in val set---
0.590

---Optimal f1 in test set---
0.559
```
