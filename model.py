import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

def normalize(df):
    for column in df:
        df = df.replace({f'{column}': {"Yes": 1, "No": 0, "No phone service": -1, "Month-to-month": 0, "One year": 1, "Two year": 2, "Electronic check": 0, "Mailed check": 1, "Bank transfer (automatic)": 2, "Credit card (automatic)": 3, "Female": 0, "Male": 1, "DSL": 1, "Fiber optic": 1, "No internet service": -1}})
    df = df.drop(columns="customerID")
    df = df[~df.apply(lambda row: any(row == ''), axis=1)]
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df.astype(float)
    df = df.dropna()
    y = df.pop(df.columns[-1])
    X = df
    return X, y

df = pd.read_csv(r'../../../../Datasets/Telco-Customer-Churn.csv') # Replace it with the path to the dataset
X, y = normalize(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

CLASSIFIERS = {SVC(): 'Support Vector Machine', 
               KNeighborsClassifier(): 'K-Nearest Neighbors', 
               DecisionTreeClassifier(): 'Decision Tree', 
               RandomForestClassifier(): 'Random Forest',  
               MLPClassifier(): 'Perceptron'}

def evaluate_classifier(clf, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test) -> float:
    model = clf.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_true=y_test, y_pred=y_pred)
    prec = precision_score(y_true=y_test, y_pred=y_pred, zero_division=0)
    rec = recall_score(y_true=y_test, y_pred=y_pred, zero_division=0)
    return acc, prec, rec

def train() -> list:
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall'])
    for index, clf in enumerate(CLASSIFIERS.keys()):
        acc, prec, rec = evaluate_classifier(clf)
        row = [CLASSIFIERS[clf], acc, prec, rec]
        results.loc[index] = row
    return results

def run():
    results = train()
    results = results.set_index('Model')
    print(results)
    print(results.columns)
    results.to_csv('results.csv', index=True)
    ax = results.plot(kind='bar')
    ax.set_xlabel('Models', ha='center', fontsize=10)
    ax.set_ylabel('Values')
    ax.set_title('Performance Metrics')
    plt.legend(loc='upper right', fontsize='small')
    plt.xticks(rotation=0, fontsize=5)
    plt.savefig('Images/plot.png')

run()