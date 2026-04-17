#Lib
import pandas as pd
from src.data_preprocessing import get_pipeline
from src.train_model import train_model
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
import pickle

# Loading Dataset :
df = pd.read_csv('data/Titanic-Dataset.csv')

X = df.drop(['Survived','PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1) # Removing the irrelevant features.
Y = df['Survived']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Spliting Data into Numeric and Category :
num_col = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_col = X_train.select_dtypes(include='str').columns

#Pipeline :
pipeline = get_pipeline(num_col, cat_col)

# we have two Option :
# 1 Use Logistic Regression model + Hyperparameter Tuning :

best_model = train_model(pipeline, X_train, Y_train)

# 2 Use improved model for better performance -- RandomForest :
# model = improve_model(X_train, Y_train)


# Evaluate the model using ('Accuracy', 'Confusion_matrix', 'Classification_report')
evaluate_model(best_model, X_test, Y_test)

# Saving Model:
with open('models/best_model.pkl','wb') as f:
    pickle.dump(best_model, f)

print('Model-Saved')


# Saving in Outputs :
# 1 Saving predictions
preds = best_model.predict(X_test)
output_df = pd.DataFrame({
    'Actual' : Y_test.values,
    'Predicted' : preds
})

output_df.to_csv('outputs/predictions.csv', index=False)


# 2 Saving metrics :
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

Acc = accuracy_score(Y_test, preds)
Cm = confusion_matrix(Y_test, preds)
Report = classification_report(Y_test, preds)

with open('outputs/metrics.txt', 'w') as f:
    f.write(f'Acc : {Acc}\n')
    f.write(f'Cm : {Cm}\n')
    f.write(f'Report : {Report}\n')


# 3 Save plotting 
import seaborn as sns
import matplotlib.pyplot as plt

# Confusion matrix --
plt.figure()
sns.heatmap(Cm, annot=True)
plt.title('Confusion_Matrix')
plt.xlabel('predicted')
plt.ylabel('Actual')

plt.savefig('outputs/confusion_matrix.png')
plt.close()

# Roc Curve --
Y_prob = best_model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(Y_test, Y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label = f'auc={roc_auc : .2f}')
plt.plot([0,1],[0,1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Roc Curve')
plt.legend()

plt.savefig('outputs/roc_curve.png')
plt.close()

print('Plot-Saved in the outputs/folder')