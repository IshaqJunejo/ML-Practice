## Day 17 - Hyperparameter Tuning

This day focuses on implementing **Hyperparameter Tuning** (`GridSeachCV`, `RandomizedSearchCV`, and `Optuna`) on multiple Machine Learning algorithms (`KNN`, `Decision Tree`, `Random Forest`, and `Gradient Boosting`) using the `diabetes` dataset.

### Results

- **K Nearest Neighbours + GridSearchCV**

Implemented classification using `K Nearest Neighbours` on the `diabetes` dataset.
I have already tried that on **Day 05**, which had achieved this performance.
```
Accuracy: 0.7727272727272727

Confusion Matrix:
[[86 13]
 [22 33]]

F1 Score: 0.6534653465346535
```

Used `GridSearchCV` for **Hyperparameter Tuning** the model, and it achieved this performance.

```
Test Accuracy: 0.7727272727272727

Confusion Matrix:
[[85 14]
 [21 34]]

F1 Score: 0.6601941747572816
```

Got only small improvement in performance (`F1-Score`).

- **Decision Tree + GridSearchCV**

I had also implemented `Decision Tree` for classification on the same `diabetes` dataset on **Day 05**, and it had such performance.
```
Accuracy: 0.7857142857142857

Confusion Matrix:
[[84 15]
 [18 37]]

F1 Score: 0.6915887850467289
```

Used `GridSearchCV` to tune the hyperparameters, and I got this result.

```
Test Accuracy:  0.7792207792207793

Confusion Matrix: 
[[83 16]
 [18 37]]

F1 Score:  0.6851851851851852
```

This time, it performed a little worse than when it was untuned.

- **Random Forest + RandomizedSearchCV**

I had also tried using `Random Forest` on the same dataset on **Day 05**.

```
Accuracy: 0.7207792207792207

Confusion Matrix:
[[77 22]
 [21 34]]

F1 Score: 0.6126126126126126
```

Tried **Hyperparameter Tuning** using the `RandomizedSearchCV`, and got this performance.

```
Test Accuracy:  0.7792207792207793

Confusion Matrix: 
[[77 22]
 [12 43]]

F1 Score:  0.7166666666666667
```

`Random Forest` got the biggest improvement in performance by **Hyperparameter Tuning**.

- **Gradient Boosting + Optuna**

I have used `GradientBoostingClassifier` before but not on the `diabetes` dataset. So for today, I only have tried it with **Hyperparameter Tuning** using `Optuna`.
```
Test Accuracy:  0.7662337662337663

Confusion Matrix: 
[[82 17]
 [19 36]]

F1 Score:  0.6666666666666666
```

I was expecting `GradientBoostingClassifier` to outperform the `Random Forest` but it didn't even compete. Turns out, the dataset was relatively small for such models.

### Unexpected Learnings

- Hyperparameter tuning doesn't neccessarily significantly improve performance of the model by just doing it once.
- Models being trained on Hyperparameter tuning do have cross-validation, yet they are still vulnerable to over-fitting. I personally noticed it with Decision Trees where the best model being returned was the one with "entropy", but when I removed the choice of "entropy", the model with "gini" was performing better.
