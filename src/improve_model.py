from sklearn.ensemble import RandomForestClassifier

# For better performance :
def improve_model(X_train, Y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=5)

    model.fit(X_train, Y_train)

    return model