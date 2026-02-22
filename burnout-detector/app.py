import pandas as pd
import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Title
st.title("🧠 Mental Burnout Detector")

# Inputs
work_hours = st.slider("Work Hours per Day", 0, 15, 8)
sleep_hours = st.slider("Sleep Hours", 0, 10, 7)
breaks = st.slider("Breaks per Day", 0, 5, 2)

stress_level = st.slider("Stress Level (1–10)", 1, 10, 5)

# Stress indicator
if stress_level <= 3:
    st.success("🟢 Low Stress")
elif stress_level <= 7:
    st.warning("🟡 Moderate Stress")
else:
    st.error("🔴 High Stress")

# SINGLE BUTTON ✅
if st.button("Check Burnout Level", key="burnout_btn"):

    input_data = pd.DataFrame({
        "work_hours": [work_hours],
        "sleep_hours": [sleep_hours],
        "stress_level": [stress_level],
        "breaks_per_day": [breaks]
    })

    pred = model.predict(input_data)[0]

    # ✅ Reasons (INSIDE button)
    reasons = []

    if work_hours > 9:
        reasons.append("Long working hours")

    if sleep_hours < 6:
        reasons.append("Low sleep duration")

    if stress_level > 7:
        reasons.append("High stress level")

    if breaks < 2:
        reasons.append("Insufficient breaks")

    # ✅ Prediction output (CORRECTLY aligned)
    if pred == 0:
        st.success("😊 Low Burnout")

    elif pred == 1:
        st.warning("😐 Medium Burnout")

    else:
        st.error("⚠️ High Burnout")

        st.markdown("### Suggestions:")
        st.write("- Take regular breaks")
        st.write("- Improve sleep schedule")
        st.write("- Reduce workload if possible")

    # ✅ Show reasons
    st.markdown("### 🔍 Why this result?")

    if reasons:
        for r in reasons:
            st.write(f"- {r}")
    else:
        st.write("👍 Your habits look balanced. No major burnout risk factors detected.")


    