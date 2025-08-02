import os
import joblib
from ucla_admission import config
from ucla_admission.data_management import load_and_prepare_data
from ucla_admission.pipeline import admission_pipeline
import random
import numpy as np
import tensorflow as tf

def set_seeds(seed=35):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seeds()

def run_training():
    """
    Trains the machine learning pipeline and saves the trained model.
    """
    print("Starting training process...")


    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print("Data loaded successfully.")

 
    print("Training the pipeline...")
    admission_pipeline.fit(X_train, y_train)
    print("Training complete.")


    os.makedirs(config.TRAINED_MODEL_DIR, exist_ok=True)


    print(f"Saving model to: {config.MODEL_SAVE_PATH}")
    joblib.dump(admission_pipeline, config.MODEL_SAVE_PATH)
    print("Model saved successfully.")


if __name__ == '__main__':
    run_training()
