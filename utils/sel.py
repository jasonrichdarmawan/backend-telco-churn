from sklearn.feature_selection import mutual_info_classif

# reference: https://stackoverflow.com/questions/44347683/how-to-set-parameters-to-score-function-in-sklearn-selectkbest
# from functools import partial
# sel_score = partial(
#     mutual_info_classif,
#     discrete_features=[False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
#     random_state=0
# )

def sel_score(X, y):
    return mutual_info_classif(
        X=X, y=y, 
        discrete_features=[False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True],
        random_state=0
    )