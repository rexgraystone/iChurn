import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import warnings
warnings.filterwarnings("ignore")

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

df = pd.read_csv(r'Telco-Customer-Churn.csv') # Replace it with the path to the dataset
X, y = normalize(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_shape, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

optimizer = Adam(learning_rate=0.00001)
early_stop = EarlyStopping(patience=10, verbose=1)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=1)

metrics = ["accuracy", "loss"]

for metric in metrics:
    plt.clf()
    plt.plot(history.history[metric], label='train')
    plt.plot(history.history[f'val_{metric}'], label='label')
    plt.legend(loc="right")
    plt.xlabel('epochs')
    plt.ylabel(metric)
    plt.title(f"CNN Model {metric.capitalize()}")
    plt.savefig(f'Images/cnn_{metric}.png')