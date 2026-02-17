import streamlit as st
import pickle
from preprocess import clean_text

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Red flag rules
red_flags = {
    "money_fast": ["earn", "instant", "daily payout", "quick money"],
    "no_experience": ["no experience", "anyone can", "easy job"],
    "contact_outside": ["gmail", "yahoo", "telegram", "whatsapp"],
    "pressure": ["urgent", "limited", "act now", "immediately"],
    "payment": ["registration fee", "pay", "deposit", "fee"]
}

def find_reasons(text):
    reasons = []
    lower = text.lower()
    for words in red_flags.values():
        for w in words:
            if w in lower:
                reasons.append(w)
    return list(set(reasons))

# Session history
if "history" not in st.session_state:
    st.session_state.history = []

# UI
st.title("🕵️ Fake Job Post Detector")
text = st.text_area("Paste Job Description Here:")

if st.button("Check"):
    cleaned = clean_text(text)
    vect = vectorizer.transform([cleaned])

    pred = model.predict(vect)[0]
    prob = model.predict_proba(vect)[0][pred]

    st.progress(int(prob * 100))

    reasons = find_reasons(text)
    st.session_state.history.append((text[:40], pred, prob))

    if pred == 1:
        st.error(f"⚠️ Fake Job (Confidence: {prob*100:.2f}%)")
        if reasons:
            st.markdown("**Why this looks suspicious:**")
            for r in reasons:
                st.write(f"- Contains scam-like phrase: `{r}`")

            highlighted = text
            for r in reasons:
                highlighted = highlighted.replace(r, f"**{r}**")
            st.markdown("### Highlighted Text")
            st.markdown(highlighted)

    else:
        st.success(f"✅ Real Job (Confidence: {prob*100:.2f}%)")
        if reasons:
            st.info("⚠️ But some risky words were found:")
            for r in reasons:
                st.write(f"- `{r}`")

# History section
st.subheader("Recent Checks")
for h in st.session_state.history[-5:]:
    label = "Fake" if h[1] == 1 else "Real"
    st.write(f"{label} | {h[2]*100:.1f}% | {h[0]}...")
