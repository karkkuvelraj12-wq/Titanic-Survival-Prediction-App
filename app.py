from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model.joblib"
FEATURES = ["pclass", "sex", "age", "fare"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Explore how a machine-learning model estimates survival probability "
    "for a passenger profile from the historical Titanic dataset."
)

@st.cache_resource
def load_model():
    """Load the trained model once and cache it for subsequent interactions."""
    return joblib.load(MODEL_PATH)

with st.sidebar:
    st.header("Passenger details")
    pclass = st.selectbox("Ticket class", options=[1, 2, 3], index=2)
    sex = st.radio("Sex", options=["female", "male"], horizontal=True)
    age = st.slider("Age", min_value=1, max_value=80, value=30)
    fare = st.slider("Fare paid", min_value=0.0, max_value=500.0, value=30.0, step=1.0)

st.caption(
    "Educational example trained on historical Titanic passenger data. "
    "It is not advice and should not be used to make real-world safety decisions."
)

if st.button("Predict survival probability", type="primary", use_container_width=True):
    if not MODEL_PATH.exists():
        st.error(
            "The model file `model.joblib` is missing. Run `python train_model.py` "
            "to train and save the model, then make sure the file is included in this project."
        )
        st.stop()

    try:
        model = load_model()
        row = pd.DataFrame(
            [[pclass, 1 if sex == "female" else 0, float(age), float(fare)]],
            columns=FEATURES,
        )
        probabilities = model.predict_proba(row)[0]
        classes = list(model.classes_)
        if 1 not in classes:
            st.error("The loaded model does not contain the expected survival class.")
            st.stop()

        chance = float(probabilities[classes.index(1)])
        st.metric("Estimated chance of survival", f"{chance:.0%}")
        if chance >= 0.5:
            st.success("The model assigns a higher probability to survival for this profile.")
        else:
            st.info("The model assigns a lower probability to survival for this profile.")

        with st.expander("Show model input"):
            display_row = row.copy()
            display_row["sex"] = "female" if sex == "female" else "male"
            st.dataframe(display_row, use_container_width=True, hide_index=True)

    except FileNotFoundError:
        st.error("Model file not found. Run `python train_model.py` and add `model.joblib`.")
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f"These inputs could not be processed. Check the model and input values. Details: {exc}")
    except Exception as exc:
        st.error(f"Prediction failed. Check that `model.joblib` matches this app. Details: {exc}")

st.divider()
st.subheader("How it works")
st.markdown(
    """
    1. Choose a passenger class, sex, age, and fare.
    2. Select **Predict survival probability**.
    3. The app sends one row to a saved Scikit-learn model and displays the estimated probability.

    The feature order must remain `pclass`, `sex`, `age`, `fare`; sex is encoded as `1 = female` and `0 = male`.
    """
)
