import pandas as pd
import pickle

# Load model
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load data
df = pd.read_csv('data/Titanic-Dataset.csv')
X = df.drop(['Survived','PassengerId', 'Name', 'Ticket', 'Cabin'],axis=1)

#test prediction
sample = X.iloc[:2]

pred = model.predict(sample)
print('predictions', pred)