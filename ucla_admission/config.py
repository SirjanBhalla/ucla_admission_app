import pathlib
import os


PACKAGE_ROOT = pathlib.Path(__file__).resolve().parent

DATA_PATH = PACKAGE_ROOT.parent.parent / "data" / "Admission.csv"

TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"
MODEL_NAME = "ucla_admission_pipeline.joblib"
MODEL_SAVE_PATH = TRAINED_MODEL_DIR / MODEL_NAME
