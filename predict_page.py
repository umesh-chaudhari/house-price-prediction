import streamlit as st
import numpy as np
import pandas as pd
import pickle


def load_all():
    model = pickle.load(open("pickles/model.pkl", "rb"))
    locations = pickle.load(open("pickles/loc.pkl", "rb"))
    locations = list(locations)
    X = pd.read_csv("xdata.csv")
    return model, locations[3:], X


model, locs, X = load_all()


def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    print(loc_index)
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]


def show_pred_page():
    st.title("Banglore House Proce Prediction")
    st.write('''### We Need Some Information About It''')
    loc = st.selectbox("Select Location", locs)
    bath = st.slider("bathroom Count", 1, 4)
    sq = st.slider("Square Feet", 50, 2000)
    bhk = st.slider("BHK Count", 1, 4)
    ok = st.button("Calculate Price")
    if ok:
        price = int(predict_price(loc, sq, bath, bhk))
        st.subheader(f"Estimated Price is: ₹ {price} lakhs")

    return None
