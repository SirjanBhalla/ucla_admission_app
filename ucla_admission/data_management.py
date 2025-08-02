import pandas as pd
from sklearn.model_selection import train_test_split
from ucla_admission import config

def load_and_prepare_data():

    data = pd.read_csv(config.DATA_PATH)


    X = data.drop(['Admit_Chance','Serial_No'], axis=1)
    y = data['Admit_Chance']


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
 
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print("Data loaded and split successfully.")
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
