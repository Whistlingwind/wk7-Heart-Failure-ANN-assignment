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

st.title("Heart Failure Assignment7")
st.markdown("_Proto_")


       
with st.sidebar:
        st.header("Configuration")
        file = st.file_uploader("Upload a file", type="csv")
        if st.button('Get data'):
           df = pd.read_csv(file)
           # This display will go away with the user's next action.
           st.write(df)
       
        if st.button('Save'):
           # This will always error.
           df.to_csv('data.csv')


#uploaded_file = pd.read_csv(r'https://raw.githubusercontent.com/Whistlingwind/wk7-Heart-Failure-ANN-assignment/main/heart_failure_clinical_records_dataset.csv')

if uploaded_file is None:
        st.info(" Upload a file through config", icon="ℹ️")
        st.stop()

df = load_data(uploaded_file)




with  st.expander("Data Preview"):
    st.markdown("_Proto_")
    st.dataframe(df)

with  st.expander("Data Preview2"):
    st.dataframe(df)
