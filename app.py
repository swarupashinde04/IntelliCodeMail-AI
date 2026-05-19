import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


# =========================
# CUSTOM UI STYLING
# =========================

st.markdown("""
<style>

body {
    background-color: #0f1117;
}

.main {
    background: linear-gradient(
        135deg,
        #0f1117,
        #1a1d29
    );
    color: white;
}

h1 {
    color: #00ffd5;
    text-align: center;
    font-size: 60px;
    text-shadow: 0px 0px 20px #00ffd5;
}

.stButton>button {

    background: linear-gradient(
        45deg,
        #00ffd5,
        #0066ff
    );

    color: white;

    border-radius: 15px;

    height: 60px;

    width: 100%;

    font-size: 20px;

    border: none;

    box-shadow: 0px 0px 20px #00ffd5;

    transition: 0.3s;
}

.stButton>button:hover {

    transform: scale(1.05);

    box-shadow: 0px 0px 40px #00ffd5;
}

textarea {

    background-color: #1e2230 !important;

    color: white !important;

    border-radius: 15px !important;
}

</style>
""", unsafe_allow_html=True)


# =========================
# TITLE
# =========================

st.title("IntelliMail AI")

st.subheader("AI-Based Spam Detection System")


# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("emails.csv")

X = data["text"]

y = data["label"]


# =========================
# TEXT VECTORIZATION
# =========================

vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)


# =========================
# TRAIN MODEL
# =========================

model = MultinomialNB()

model.fit(X_vectorized, y)


# =========================
# USER INPUT
# =========================

email_input = st.text_area(
    "Enter Email Message"
)


# =========================
# ANALYZE BUTTON
# =========================

if st.button("Analyze Email"):

    # Convert input text
    input_vector = vectorizer.transform(
        [email_input]
    )

    # Prediction
    prediction = model.predict(
        input_vector
    )

    # Probability
    probability = model.predict_proba(
        input_vector
    )

    spam_probability = probability[0][
        list(model.classes_).index("spam")
    ]

    spam_percent = spam_probability * 100


    # =========================
    # OUTPUT RESULT
    # =========================

    if prediction[0] == "spam":

        st.error(
            "Spam Email Detected!"
        )

    else:

        st.success(
            "Safe Email"
        )


    # =========================
    # SPAM PROBABILITY
    # =========================

    st.subheader("Spam Probability")

    st.write(
        f"{spam_percent:.2f}%"
    )

    st.progress(
        int(spam_percent)
    )


    # =========================
    # RISK LEVEL
    # =========================

    st.subheader("Risk Level")

    if spam_percent < 40:

        st.success("LOW")

    elif spam_percent < 70:

        st.warning("MEDIUM")

    else:

        st.error("HIGH")


    # =========================
    # SUSPICIOUS KEYWORDS
    # =========================

    suspicious_words = [
        "free",
        "win",
        "money",
        "urgent",
        "click",
        "offer",
        "prize",
        "bank",
        "limited",
        "reward"
    ]

    detected_words = []

    for word in suspicious_words:

        if word in email_input.lower():

            detected_words.append(word)

    if detected_words:

        st.subheader(
            "Suspicious Keywords"
        )

        for word in detected_words:

            st.warning(f"⚠ {word}")


    # =========================
    # AI RECOMMENDATION
    # =========================

    st.subheader("AI Recommendation")

    if prediction[0] == "spam":

        st.warning(
            "Avoid clicking suspicious links or sharing personal information."
        )

    else:

        st.success(
            "This email appears relatively safe."
        )
