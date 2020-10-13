# Baseline methods

Method   | Micro f1
-------- | -------
SciSpacy | 0.36
MTI      | 0.49

# Model inventory

Version   | Approach    | Micro f1 in Pubs | Micro f1 in Grants | Description
--------- | ----------- | ---------------- | ------------------ | -------
2020.07.0 | tfidf-sgd   | 0.56             | 0.69               | tfidf-svm, bigrams, regularisation of 1e-9, removal of words appearing less than 5 times

## 2020.07.0

```
              precision    recall  f1-score   support
   micro avg       0.72      0.46      0.56   4153163
   macro avg       0.64      0.35      0.44   4153163
weighted avg       0.69      0.46      0.54   4153163
 samples avg       0.61      0.52      0.53   4153163
```
