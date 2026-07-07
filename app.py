import streamlit as st
import pickle
import numpy as np

# ---------------- Page Configuration ----------------

st.set_page_config(
    page_title="Predictive Healthcare Assistant",
    page_icon="🏥",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;color:#0E76A8;'>🏥 Predictive Healthcare Assistant</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;color:gray;'>AI Powered Disease Prediction System</h4>",
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------- Sidebar ----------------

with st.sidebar:

    st.title("🏥 Healthcare AI")

    disease = st.radio(
        "Select Disease",
        [
            "❤️ Heart Disease",
            "🍬 Diabetes",
            "🎗 Breast Cancer"
        ]
    )

    st.markdown("---")
    st.info("Machine Learning Based Prediction")

# =====================================================
# HEART DISEASE
# =====================================================

if disease == "❤️ Heart Disease":

    model = pickle.load(open("heart_disease_model.pkl", "rb"))

    st.subheader("❤️ Heart Disease Prediction")

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age",20,100,50)
        gender = st.selectbox("Gender",["Female","Male"])
        bp = st.number_input("Blood Pressure",80,200,120)

    with col2:
        chol = st.number_input("Cholesterol",100,500,200)
        heart_rate = st.number_input("Heart Rate",60,220,150)

    sex = 1 if gender=="Male" else 0

    # Default Values
    cp=0
    fbs=0
    restecg=1
    exang=0
    oldpeak=1.0
    slope=1
    ca=0
    thal=2

    if st.button("Predict Heart Disease"):

        data=np.array([[age,sex,cp,bp,chol,fbs,
                        restecg,heart_rate,exang,
                        oldpeak,slope,ca,thal]])

        prediction=model.predict(data)

        if prediction[0]==1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

# =====================================================
# DIABETES
# =====================================================

elif disease=="🍬 Diabetes":

    model=pickle.load(open("diabetes_model.pkl","rb"))

    st.subheader("🍬 Diabetes Prediction")

    col1,col2=st.columns(2)

    with col1:

        age=st.slider("Age Group",1,13,6)
        gender=st.selectbox("Gender",["Female","Male"])
        bmi=st.number_input("BMI",10.0,50.0,25.0)

    with col2:

        highbp=st.selectbox("High Blood Pressure",["No","Yes"])
        highchol=st.selectbox("High Cholesterol",["No","Yes"])

    sex=1 if gender=="Male" else 0
    highbp=1 if highbp=="Yes" else 0
    highchol=1 if highchol=="Yes" else 0

    # Default values

    cholcheck=1
    smoker=0
    stroke=0
    heart=0
    activity=1
    fruits=1
    veggies=1
    alcohol=0
    healthcare=1
    nodoc=0
    genhlth=2
    menthlth=0
    physhlth=0
    diffwalk=0
    education=4
    income=5

    if st.button("Predict Diabetes"):

        data=np.array([[highbp,highchol,cholcheck,bmi,
                        smoker,stroke,heart,
                        activity,fruits,veggies,
                        alcohol,healthcare,nodoc,
                        genhlth,menthlth,physhlth,
                        diffwalk,sex,age,
                        education,income]])

        prediction=model.predict(data)

        if prediction[0]==0:
            st.success("✅ No Diabetes")

        elif prediction[0]==1:
            st.warning("⚠️ Prediabetes")

        else:
            st.error("❌ Diabetes Detected")

# =====================================================
# BREAST CANCER
# =====================================================

else:

    model=pickle.load(open("breast_cancer_model.pkl","rb"))

    st.subheader("🎗 Breast Cancer Prediction")

    col1,col2=st.columns(2)

    with col1:

        radius=st.number_input("Mean Radius",1.0,30.0,14.0)
        texture=st.number_input("Mean Texture",1.0,40.0,20.0)
        perimeter=st.number_input("Mean Perimeter",20.0,200.0,90.0)

    with col2:

        area=st.number_input("Mean Area",100.0,3000.0,600.0)
        smoothness=st.number_input("Mean Smoothness",0.01,0.30,0.10)

    sample=np.zeros((1,30))

    sample[0,0]=radius
    sample[0,1]=texture
    sample[0,2]=perimeter
    sample[0,3]=area
    sample[0,4]=smoothness

    if st.button("Predict Breast Cancer"):

        prediction=model.predict(sample)

        if prediction[0]==1:
            st.success("✅ Benign (Low Risk)")
        else:
            st.error("⚠️ Malignant (High Risk)")

# ---------------- Footer ----------------

st.markdown("---")

st.info(
"""
💡 Health Tips

✔ Exercise Regularly

✔ Eat Healthy Food

✔ Drink Plenty of Water

✔ Avoid Smoking

✔ Regular Medical Checkups
"""
)

st.markdown("---")

st.caption("© 2026 Predictive Healthcare Assistant | Developed by Sanuja")