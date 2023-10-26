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

st.title("Heart Failure Assignment8")
st.markdown("_Proto_")


       
with st.sidebar:
        st.header("Configuration")
        file = st.file_uploader("Upload a file", type="csv")
        if st.button('Get data'):
           df = pd.read_csv(file)
           
       
        if st.button('Save'):
           # This will always error.
           df.to_csv('data.csv')






with  st.expander("Data Preview"):
    st.markdown("_Proto_")
    st.dataframe(df)

with  st.expander("Data Preview2"):
    st.dataframe(df)
