from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def evaluate_model(best_model, X_test, Y_test):

    Y_pred = best_model.predict(X_test) # Testing the model on Unseen data

# Evaluation :  
    Acc = accuracy_score(Y_test, Y_pred)
    Cm = confusion_matrix(Y_test, Y_pred)
    Report = classification_report(Y_test, Y_pred)

    print('Accuracy :', Acc)
    print('Confusion_matrix :',Cm)
    print('Classification_report :', Report)

