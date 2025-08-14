# **UCLA ADMISSION PREDICTOR**
A user-friendly web app that predicts your probability of admission to UCLA's graduate programs based on academic profile and research experience.

### STREAMLIT APP LINK:
https://uclaadmissionapp-whsogjiak6lip8pxk2jd9r.streamlit.app

### OVERVIEW:

Input: GRE, TOEFL, University Rating, SOP, LOR, CGPA, Research Experience

Output: Probability of admission indicated by a category (High Chance, Maybe, Low Chance)

Built with: Python, Streamlit, scikit-learn, TensorFlow, Keras

----------------------------------------------------------------------------------------------------------------------------

### PROJECT STRUCTURE:
   ```markdown
ucla_admission_app/
├── app.py
├── ucla_admission/
│    ├── __init__.py
│    ├── config.py
│    ├── data_management.py
│    ├── pipeline.py
│    ├── trained_models/
│    │    └── ucla_admission_pipeline.joblib
├── requirements.txt
├── README.md
```
----------------------------------------------------------------------------------------------------------------------------

### HOW TO RUN THE APP:

##### 1. Clone the repo and change directory:
    ```terminal
    git clone <your-repo-link>
    cd ucla_admission_app
    ```

##### 2. Create & activate a virtual environment (recommended):
    ```terminal
    python -m venv .venv
    source .venv/bin/activate        # On Windows: .venv\Scripts\activate
    ```

##### 3. Install dependencies:
    ```terminal
    pip install -r requirements.txt
    ```

##### 4. Start the Streamlit app:
    ```terminal
    streamlit run app.py
    ```

##### 5. Open the browser link

----------------------------------------------------------------------------------------------------------------------------

### FEATURES:

1. Clean interface with clear input guidance.

2. Explains what values are required for each field.

3. Shows both the predicted percentage and an easy-to-understand category.

4. Friendly notes and links to official UCLA admissions.

----------------------------------------------------------------------------------------------------------------------------

### MODEL INFO:

1. Model type: Keras neural network regression

2. Input features: GRE Score, TOEFL Score, University Rating, SOP, LOR, CGPA, Research Experience

3. Output: Predicted probability of admission (0–1)

4. Best reproducible R²: ~0.74

5. Random seed: Set for reproducible results

----------------------------------------------------------------------------------------------------------------------------

*See requirements.txt for exact versions.*


