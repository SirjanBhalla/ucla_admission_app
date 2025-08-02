import streamlit as st
import joblib
import numpy as np
from ucla_admission import config


model = joblib.load(config.MODEL_SAVE_PATH)


st.set_page_config(page_title="UCLA Admission Predictor", page_icon=":mortar_board:", layout="centered")


st.title("UCLA Admission Predictor")
st.markdown(
    "Welcome! Please enter your stats to check the probability for your admission in UCLA. \n\n*Built with care by an Algonquin Student*"
)

st.divider()


st.markdown("### Input details (please enter values within these ranges):")
st.markdown("""
- **GRE Score:** out of 340  
- **TOEFL Score:** out of 120  
- **University Rating:** 1 to 5  
- **SOP:** Statement of Purpose strength (1 to 5)  
- **LOR:** Letter of Recommendation strength (1 to 5)  
- **CGPA:** GPA out of 10  
- **Research:** Yes (1) or No (0)
""")


col1, col2, col3 = st.columns(3)

with col1:
    gre = st.number_input("GRE Score", min_value=0, max_value=340, step=1)
    toefl = st.number_input("TOEFL Score", min_value=0, max_value=120, step=1)
    rating = st.selectbox("University Rating", [1, 2, 3, 4, 5])

with col2:
    sop = st.slider("Statement of Purpose strength (SOP)", min_value=1.0, max_value=5.0, step=0.5)
    lor = st.slider("Letter of Recommendation strength (LOR)", min_value=1.0, max_value=5.0, step=0.5)

with col3:
    cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, step=0.01, format="%.2f")
    research = st.radio("Research Experience", ["No", "Yes"])

st.divider()

if st.button("Predict Admission Chance"):

    X = np.array([[gre, toefl, rating, sop, lor, cgpa, 1 if research == "Yes" else 0]])

    prediction = model.predict(X)[0]
    prediction = max(0, min(prediction, 1))

    if prediction >= 0.8:
        label = ":star: **High Chance**"
        color = "green"
    elif prediction >= 0.6:
        label = "🟡 **Maybe**"
        color = "gold"
    else:
        label = ":x: **Low Chance**"
        color = "red"

    st.markdown(f"### Admission Probability:")
    st.markdown(f"<span style='color:{color}; font-size:1.3em'>{label}</span>", unsafe_allow_html=True)
    
    st.info(
    "Note: This is just a predictive estimate based on historical data. The results may not always be accurate. "
    "Please refer to official UCLA admission guidelines for more information. [UCLA Website](https://admission.ucla.edu/)", icon="ℹ️"
)
