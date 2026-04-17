from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


def train_model(pipeline, X_train, Y_train):

# Connecting Pipeline and Model :
    model = Pipeline([
        ('preprocessor', pipeline),
        ('model', LogisticRegression(class_weight='balanced')) # We balanced the 2 classes
    ])

# Tuning the model:
    param_grid = {

        'model__C' : [0.001, 0.1, 1, 10, 100],
        'model__penalty' : ['l2'],
        'model__solver' : ['lbfgs']
    }

    grid = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='recall'
    )

    grid.fit(X_train, Y_train)

    print('Best_parameter :',grid.best_params_)
    print('Best_score :',grid.best_score_)

    best_model = grid.best_estimator_


    return best_model











    # cv_scores = cross_val_score(
    #     model,
    #     X_train,
    #     Y_train,
    #     cv=5,
    #     scoring='recall'
    # )

    # print('Cross_val_score :', cv_scores)
    # print('Cv_score_mean :', cv_scores.mean())

    # model.fit(X_train, Y_train) # Training the model

    # return model