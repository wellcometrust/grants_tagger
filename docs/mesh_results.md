# Baseline methods

Method   | Micro f1
-------- | -------
SciSpacy | 0.36
MTI      | 0.49

# Model inventory

Version   | Approach    | Micro f1 in Pubs | Micro f1 in Grants | Model size | Train time | Description
--------- | ----------- | ---------------- | ------------------ | -----------| ---------- | ----------
2020.07.0 | tfidf-sgd   | 0.56             | 0.69               | ~70GB      | | tfidf-svm, bigrams, regularisation of 1e-9, removal of words appearing less than 5 times
2020.9.0  | cnn         | 0.64             |                    | 2.5GB      | | 1e-3 lr, 0.1 dr, 1e-7 l2, 4 layers, attention
2020.9.1  | cnn         | 0.63             |                    |            | | 1e-3 lr, 0.1 dr, **1e-8 l2**, 4 layers, attention
2020.9.2  | cnn         | 0.64             |                    |            | | 1e-3 lr, 0.1 dr, 1e-7 l2, 4 layers, ~~attention~~
2020.9.3  | cnn         | 0.57             |                    |            | | 1e-3 lr, 0.1 dr, 1e-7 l2, **0 layers**, attention
2020.9.4  | cnn         | 0.62             |                    |            | | 1e-3 lr, **0 dr**, 1e-7 l2, 4 layers, attention
2020.9.5  | cnn         | 0.63             |                    |            | | 1e-3 lr, 0.1 dr, **1e-9 l2**, 4 layers, attention
2020.9.6  | cnn         | 0.64             |                    |            | | 1e-3 lr, 0.1 dr, 1e-7 l2, 8 layers, attention
2021.3.0  | cnn         | 0.63             | 0.68               | 765MB      | | same as 2020.09.0 but with transformers as tokenizers_library
2021.6.0  | mesh-cnn    | 0.62             | 0.66               |            | | same as 2021.3.0 but with mesh 2021 data that contains COVID
2021.9.0  | mesh-xlinear| 0.53             | 0.56 (disease mesh)| 4GB        | 2h50m | mesh-xlinear with cluster chain
2021.10.0 | mesh-xlinear| 0.53             | 0.55 (disease mesh)| 6GB        | 1h19m | mesh-xlinear with parallel tf-idf

## 2020.07.0

```
              precision    recall  f1-score   support
   micro avg       0.72      0.46      0.56   4153163
   macro avg       0.64      0.35      0.44   4153163
weighted avg       0.69      0.46      0.54   4153163
 samples avg       0.61      0.52      0.53   4153163
```

## 2020.09.0

```
   micro avg       0.77      0.54      0.64   4153163
   macro avg       0.64      0.41      0.48   4153163
weighted avg       0.74      0.54      0.61   4153163
 samples avg       0.72      0.62      0.64   4153163
```

## 2020.09.1

```
   micro avg       0.77      0.54      0.63   4153163
   macro avg       0.66      0.42      0.49   4153163
weighted avg       0.74      0.54      0.61   4153163
 samples avg       0.72      0.62      0.63   4153163
```

## 2020.09.2

```
   micro avg       0.75      0.55      0.64   4153163
   macro avg       0.63      0.42      0.48   4153163
weighted avg       0.73      0.55      0.61   4153163
 samples avg       0.72      0.63      0.64   4153163
```

## 2020.09.3

```
   micro avg       0.75      0.46      0.57   4153163
   macro avg       0.59      0.33      0.40   4153163
weighted avg       0.70      0.46      0.54   4153163
 samples avg       0.65      0.54      0.56   4153163
```

## 2020.09.4

```
   micro avg       0.75      0.54      0.62   4153163
   macro avg       0.63      0.41      0.48   4153163
weighted avg       0.72      0.54      0.60   4153163
 samples avg       0.71      0.61      0.63   4153163
```

## 2020.09.5
```
   micro avg       0.77      0.53      0.63   4153163
   macro avg       0.67      0.42      0.49   4153163
weighted avg       0.74      0.53      0.60   4153163
 samples avg       0.72      0.61      0.63   4153163
```

## 2020.09.6
```
   micro avg       0.76      0.56      0.64   4153163                        
   macro avg       0.64      0.42      0.49   4153163                        
weighted avg       0.73      0.56      0.62   4153163                        
 samples avg       0.73      0.64      0.65   4153163
```

## 2021.03.0
```
   micro avg       0.77      0.54      0.63   4153163
   macro avg       0.64      0.40      0.47   4153163
weighted avg       0.74      0.54      0.61   4153163
 samples avg       0.72      0.62      0.63   4153163
```

## 2021.06.0
```
   micro avg       0.78      0.52      0.62   4357732
   macro avg       0.61      0.36      0.43   4357732
weighted avg       0.75      0.52      0.59   4357732
 samples avg       0.72      0.60      0.62   4357732
```

## 2021.09.0
```
   micro avg       0.74      0.41      0.53   1975110
   macro avg       0.49      0.26      0.32   1975110
weighted avg       0.68      0.41      0.49   1975110
 samples avg       0.74      0.42      0.52   1975110
```
