## Import streamlit to learn more visualisation tools, cause learning is cool.
## This took ages to setup...!

import streamlit as st
from streamlit.connections import ExperimentalBaseConnection
import duckdb


# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style
import numpy as np 
import plotly.express as px
import plotly.figure_factory as ff
## Import required libraries
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score
import pandas as pd
import streamlit as st

st.set_page_config(
       page_title="Heart Failure ANN",
       page_icon="	:heartpulse:",
       layout="wide"
)




@st.cache_data
def load_data(file):

        data = pd.read_csv(file)
        return data
    
with st.sidebar:
        st.header("Configuration")
        uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is None:
        st.info(" Upload a file through config", icon="ℹ️")
        st.stop()

df = load_data(uploaded_file)
# Check if you've already initialized the data
if 'df' not in st.session_state:
    # Save the data to session state
    st.session_state.df = df


def intro():
    import streamlit as st
    import pandas as pd
    
    st.title("Assignment Wk7: Heart Failure Prediction: SVM and ANN")
    multi = '''Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide. Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
                    
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.
    '''
    st.markdown(multi)
    st.sidebar.success("Select a demo above.")

    st.image("https://raw.githubusercontent.com/Whistlingwind/wk7-Heart-Failure-ANN-assignment/main/Heart_Failure.jpg", channels="RGB", output_format="auto")

    st.title("Core Areas of Interest")
    st.markdown(":bar_chart:  Attempt to learn & build re-usable interface, that is publishable, shareable and editable online via Github, as well as able to display animated plots and complex concepts")
    st.markdown(":chart_with_downwards_trend: Review Data Wrangling for any potential issues blocking analysis or predictive accuracy")
    st.markdown(":chart_with_upwards_trend: Train & Test a Artificial Neural Network")
    st.markdown(":sparkle: Bonus: Attempt to create data entry for users to be able to generate predictions from data entry")









def model_building():
    import warnings
    warnings.filterwarnings('ignore')
    import streamlit as st
    import numpy as np
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    from sklearn import preprocessing
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm 
    from keras.layers import Dense, BatchNormalization, Dropout, LSTM
    from keras.models import Sequential
    from keras import callbacks
    import sklearn.metrics as metrics
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler  
    from sklearn.neighbors import KNeighborsClassifier
    from PIL import Image 



    df = st.session_state.df
    st.markdown(f'# {list(page_names_to_funcs.keys())[2]}')
    st.write(
        """
        We will attempt to build three models, "Decision Trees", "Neural Networks", "K-Nearest Neighbours", below we are doing a quick data check to review any anomalies.
"""
    )


    # Defining independent and dependent attributes in training and test sets
    X=df.drop(["DEATH_EVENT"],axis=1)
    y=df["DEATH_EVENT"]
    st.write("The data has been prepocessed below with Standard Scaler")
    # Setting up a standard scaler for the features and analyzing it thereafter
    col_names = list(X.columns)
    s_scaler = preprocessing.StandardScaler()
    X_scaled= s_scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=col_names)   
    X_scaled.describe().T


    st.write("I've reviewed Box Plots across the feature set to catch any outliers, but with the scaling, none seem to be a problem.")
    @st.cache_data
    def get_chart_5205(use_container_width: bool):
        import altair as alt
        



        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Age", "creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium", "time"])

        with tab1:
                chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    x='DEATH_EVENT',
                    y="age"
                )
                st.altair_chart(chart, theme="streamlit", use_container_width=True)
        with tab2:
                chart2 = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    x='DEATH_EVENT',
                    y="creatinine_phosphokinase"
                )
                st.altair_chart(chart2, theme="streamlit", use_container_width=True)

        with tab3:
                chart3 = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    x='DEATH_EVENT',
                    y="ejection_fraction"
                )
                st.altair_chart(chart3, theme="streamlit", use_container_width=True)
        with tab4:
                chart4 = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    x='DEATH_EVENT',
                    y="platelets"
                )
                st.altair_chart(chart4, theme="streamlit", use_container_width=True)
        with tab5:
                chart5 = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    x='DEATH_EVENT',
                    y="serum_creatinine"
                )
                st.altair_chart(chart5, theme="streamlit", use_container_width=True)
        with tab6:
                chart6 = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    x='DEATH_EVENT',
                    y="serum_sodium"
                )
                st.altair_chart(chart6, theme="streamlit", use_container_width=True)
        with tab7:
                chart7 = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                    x='DEATH_EVENT',
                    y="time"
                )
                st.altair_chart(chart7, theme="streamlit", use_container_width=True)


    get_chart_5205(1)

    #spliting variables into training and test sets
    X_train, X_test, y_train,y_test = train_test_split(X_scaled,y,test_size=0.30,random_state=25)

    # Instantiating the SVM algorithm 
    model1=svm.SVC()

    # Fitting the model 
    model1.fit (X_train, y_train)

    # Predicting the test variables
    y_pred = model1.predict(X_test)

    # Getting the score 
    model1.score (X_test, y_test)

    
    st.write("Checking classification report (since there was biasness in target features)")
    st.dataframe(classification_report(y_test, y_pred, output_dict=True))

    features= df[["age", "creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium", "time"]].values
    labels = df['DEATH_EVENT'].values
    
    X_train,X_test, y_train, y_test = train_test_split(features, labels, train_size=0.7, random_state=1)
    st.write("Testing Algorithm outputs, Decision Tree and SVM to start with.")
    alg = ['Decision Tree', 'Support Vector Machine']
    classifier = st.selectbox('Which algorithm?', alg)
    if classifier=='Decision Tree':
        dtc = DecisionTreeClassifier()
        dtc.fit(X_train, y_train)
        acc = dtc.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_dtc = dtc.predict(X_test)
        cm_dtc=confusion_matrix(y_test,pred_dtc)
        st.write('Confusion matrix: ', cm_dtc)
    
        
    elif classifier == 'Support Vector Machine':
        svm=SVC()
        svm.fit(X_train, y_train)
        acc = svm.score(X_test, y_test)
        st.write('Accuracy: ', acc)
        pred_svm = svm.predict(X_test)
        cm=confusion_matrix(y_test,pred_svm)
        st.write('Confusion matrix: ', cm)


        
    # Basic preprocessing required for all the models.  
    def preprocessing(df):
        # Assign X and y
        # Defining independent and dependent attributes in training and test sets
        X=df.drop(["DEATH_EVENT"],axis=1)
        y=df["DEATH_EVENT"]


        # 1. Splitting X,y into Train & Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
        return X_train, X_test, y_train, y_test


    # Training Decission Tree for Classification
    @st.cache(suppress_st_warning=True)
    def decisionTree(X_train, X_test, y_train, y_test):
        # Train the model
        tree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
        tree.fit(X_train, y_train)
        y_pred = tree.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred) * 100
        report = classification_report(y_test, y_pred,output_dict=True)
        #report = classification_report(y_test, y_pred)
        
        return score, report, tree

    # Training Neural Network for Classification.
    @st.cache(suppress_st_warning=True)
    def neuralNet(X_train, X_test, y_train, y_test):
        # Scalling the data before feeding it to the Neural Network.
        scaler = StandardScaler()  
        scaler.fit(X_train)  
        X_train = scaler.transform(X_train)  
        X_test = scaler.transform(X_test)
        # Instantiate the Classifier and fit the model.
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score1 = metrics.accuracy_score(y_test, y_pred) * 100
        report = classification_report(y_test, y_pred,output_dict=True)
        
        return score1, report, clf

    # Training KNN Classifier
    @st.cache(suppress_st_warning=True)
    def Knn_Classifier(X_train, X_test, y_train, y_test):
        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = metrics.accuracy_score(y_test, y_pred) * 100
        report = classification_report(y_test, y_pred,output_dict=True)

        return score, report, clf


    # Accepting user data for predicting its Member Type
    def accept_user_data():
        age = st.text_input("Enter age: ")
        anaemia = st.text_input("Anaemia |Enter Boolean 0 or 1: ")
        creatinine_phosphokinase = st.text_input("Creatinine_phosphokinase | Enter 1-8000: ")
        diabetes = st.text_input("Diabetes | Enter Boolean 0 or 1: ")
        ejection_fraction = st.text_input("Ejection_fraction | Enter 1-80: ")
        high_blood_pressure = st.text_input("High_blood_pressure | Enter Boolean 0 or 1: ")
        platelets = st.text_input("Platelets | Enter 20000 - 850000: ")
        serum_creatinine = st.text_input("Serum_creatinine | Enter 0.1 - 1: ")
        sex = st.text_input("Sex | Enter Boolean 0 or 1: ")
        smoking = st.text_input("Smoking | Enter Boolean 0 or 1: ")
        time = st.text_input("Enter day of visit | 1-365: ")






        user_prediction_data = np.array([age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,sex,smoking,time]).reshape(1,-1)

        return user_prediction_data


    def main():
        
        #data = df
        #X_train, X_test, y_train, y_test

        # Insert Check-Box to show the snippet of the data.
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")	
            st.write(df.head())


        # ML Section
        choose_model = st.sidebar.selectbox("Choose the ML Model",
            ["NONE","Decision Tree", "Neural Network", "K-Nearest Neighbours"])

        if(choose_model == "Decision Tree"):
            score, report, tree = decisionTree(X_train, X_test, y_train, y_test)
            st.text("Accuracy of Decision Tree model is: ")
            st.write(score,"%")
            st.text("Report of Decision Tree model is: ")
            #st.write(report)
            st.dataframe(report)

            try:
                if(st.checkbox("Want to predict on your own Input? Well its WIP, and I ran out of time, SORRY! 	:joy: ")):
                    user_prediction_data = accept_user_data() 		
                    pred = tree.predict(user_prediction_data)

            except:
                pass

        elif(choose_model == "Neural Network"):
            score, report, clf = neuralNet(X_train, X_test, y_train, y_test)
            st.text("Accuracy of Neural Network model is: ")
            st.write(score,"%")
            st.text("Report of Neural Network model is: ")
            #st.write(report)
            st.dataframe(report)

            try:
                if(st.checkbox("Want to predict on your own Input? Well its WIP, and I ran out of time, SORRY! 	:joy: ")):
                    user_prediction_data = accept_user_data()
                    scaler = StandardScaler()  
                    scaler.fit(X_train)  
                    user_prediction_data = scaler.transform(user_prediction_data)	
                    pred = clf.predict(user_prediction_data)

            except:
                pass

        elif(choose_model == "K-Nearest Neighbours"):
            score, report, clf = Knn_Classifier(X_train, X_test, y_train, y_test)
            st.text("Accuracy of K-Nearest Neighbour model is: ")
            st.write(score,"%")
            st.text("Report of K-Nearest Neighbour model is: ")
            #st.write(report)
            st.dataframe(report)

            try:
                if(st.checkbox("Want to predict on your own Input? Well its WIP, and I ran out of time, SORRY! 	:joy: ")):
                    user_prediction_data = accept_user_data() 		
                    pred = clf.predict(user_prediction_data)

            except:
                pass
        
        




    if __name__ == "__main__":
        main()


def data_Wrangling():
    import streamlit as st
    import time
    import numpy as np
    from PIL import Image
    df = st.session_state.df
    st.markdown(f'# {list(page_names_to_funcs.keys())[1]}')
    st.write(
        """
        Reviewing the data for any inconsistencies, NaN values and typical deviation patterns.
"""
    )

    with    st.expander("Data Preview"):
            st.markdown("_All of the data appears workable, although both platelets and creatinine_phosphokinase have heavily skewed data._")
            st.dataframe(df)

    st.markdown("Observation across the entire dataset shows a skew towards one side, when looking at the Death Events. With a significant amounts of deaths present.")

    death_data = duckdb.sql(
         f"""
         WITH death_events AS(
         Select SUM(DEATH_EVENT) AS Deaths, SUM(CASE WHEN DEATH_EVENT = 0 
                THEN 1 
                ELSE 0 END) AS Alive
         FROM df
         GROUP BY DEATH_EVENT
         )
         SELECT SUM(Deaths) AS Deaths, SUM(Alive) AS Alive
         FROM death_events

    """
    ).df()

    
    
    #st.dataframe(death_data)
  
    st.bar_chart(death_data)

    st.markdown("Review whether there is a bias across sex, we can see the sex (1) scores markedly higher.")
    st.bar_chart(df, x="sex",y="DEATH_EVENT", color="sex")
    st.markdown("Reviewing the data, we can see that the two skews mentioned are present, the rest of the features have a normal distribution.")
    df.describe().T




page_names_to_funcs = {
    "—": intro,
    "Data Wrangling": data_Wrangling,
    "Model Building": model_building
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
