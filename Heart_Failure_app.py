## Import streamlit to learn more visualisation tools, cause learning is cool.
## This took ages to setup...!

import streamlit as st
import pandas as pd
import streamlit as st

st.set_page_config(
       page_title="Heart Failure ANN",
       page_icon="	:heartpulse:",
       layout="wide"
)

st.title("Heart Failure Assignment")
st.markdown("_Proto_")

@st.cache_data
def load_data(file):
        data = pd.read_csv(file)

        return data
with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.sidebar.file_uploader("Choose a file")
        st.button(label="Use Default Data", key="btn_default_file")
#df = load_data("./heart_failure_clinical_records_dataset.csv")

if st.session_state.get("btn_default_file"):
        uploaded_file = load_data(r'https://raw.githubusercontent.com/Whistlingwind/wk7-Heart-Failure-ANN-assignment/main/heart_failure_clinical_records_dataset.csv')


if uploaded_file is None:
        st.info(" Upload a file through config", icon="ℹ️")
        st.stop()

df = load_data(uploaded_file)




with  st.expander("Data Preview"):
    st.markdown("_Proto_")
    st.dataframe(df)

with  st.expander("Data Preview2"):
    st.dataframe(df)
