import joblib
from ucla_admission import config
from ucla_admission.data_management import load_and_prepare_data
from sklearn.metrics import mean_absolute_error, r2_score

def make_prediction():
    """
    Loads the trained pipeline, makes predictions on the test set,
    and evaluates the model's performance.
    """

    X_train, X_test, y_train, y_test = load_and_prepare_data()

 
    pipeline_path = config.MODEL_SAVE_PATH
    trained_pipeline = joblib.load(filename=pipeline_path)

  
    predictions = trained_pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2): {r2:.4f}")
    print("------------------------")

if __name__ == '__main__':
    make_prediction()
