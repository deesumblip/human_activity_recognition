# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 11:13:28 2020

@author: Dilini
"""
from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

model = load_model('FinalLGBMModel12Oct2020')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df['Label'][0]
    return predictions

def run():

    st.sidebar.title("Human Movement Prediction App")
    add_selectbox = st.sidebar.selectbox(
        "How would you like to predict?",
        ("Manual", "Batch"))


#   st.sidebar.success('https://www.pycaret.org') # CHANGE LATER
    
    st.title("Explore Motion Sensor Data")
    st.markdown('Select Batch from sidebar dropdown menu to view batch predictions below')

    if add_selectbox == 'Manual':

        st.sidebar.info('Move the sliders and click Predict to generate an Activity prediction')
        tGravityAccmeanX = st.sidebar.slider('tGravityAccmeanX', -1.0, 1.0, 0.5, 0.01)
        tGravityAccCorrelationXY = st.sidebar.slider('tGravityAccCorrelationXY', -1.0, 1.0, 0.5, 0.01)
        tGravityAccCorrelationXZ = st.sidebar.slider('tGravityAccCorrelationXZ', -1.0, 1.0, 0.5, 0.01)
        tGravityAccCorrelationYZ = st.sidebar.slider('tGravityAccCorrelationYZ', -1.0, 1.0, 0.5, 0.01)
        angletBodyGyroMeanGravityMean = st.sidebar.slider('angletBodyGyroMeanGravityMean', -1.0, 1.0, 0.5, 0.01)
        subject = st.sidebar.number_input('subject', min_value=1, max_value=30)

        output=""

        input_dict = {'tGravityAccmeanX' : tGravityAccmeanX, 
                      'tGravityAccCorrelationXY' : tGravityAccCorrelationXY, 
                      'tGravityAccCorrelationXZ' : tGravityAccCorrelationXZ,
                      'tGravityAccCorrelationYZ' : tGravityAccCorrelationYZ,
                      'angletBodyGyroMeanGravityMean' : angletBodyGyroMeanGravityMean, 
                      'subject' : subject,}
        input_df = pd.DataFrame([input_dict])
        

        if st.sidebar.button("Predict"):
            output = predict(model=model, input_df=input_df)
            output = str(output)
        
        st.sidebar.success('OUTPUT: {}'.format(output))

## EDIT THIS PORTION FOR PLOTLY
    dataset_name = st.selectbox("Select dataset to view data", ("Unseen Data",))

    if dataset_name == "Unseen Data":

        data_location = ('data_unseen.csv')

        @st.cache
        def load_data():
            data = pd.read_csv(data_location)
            return data

        df = load_data()

        # show data on streamlit
        st.write(df)

        # show data in plotly chart
        graph_selectbox = st.selectbox("Select a motion sensor from data to view graph", ('tGravityAccmeanX','tGravityAccCorrelationXY',
                                                                                                 'tGravityAccCorrelationXZ','tGravityAccCorrelationYZ','angletBodyGyroMeanGravityMean'))

        if graph_selectbox == 'tGravityAccmeanX':
            fig = px.scatter_3d(df, x='subject', y='timepoint', z='tGravityAccmeanX',
                                color='Activity', height=700, width=800)
            fig.update_traces(mode='markers', marker_size=4, opacity=0.5)
            st.plotly_chart(fig)
        elif graph_selectbox == 'tGravityAccCorrelationXY':
            fig2 = px.scatter_3d(df, x='subject', y='timepoint', z='tGravityAccCorrelationXY',
                                color='Activity', height=700, width=800)
            fig2.update_traces(mode='markers', marker_size=4, opacity=0.5)
            st.plotly_chart(fig2)
        elif graph_selectbox == 'tGravityAccCorrelationXZ':
            fig3 = px.scatter_3d(df, x='subject', y='timepoint', z='tGravityAccCorrelationXZ',
                                color='Activity', height=700, width=800)
            fig3.update_traces(mode='markers', marker_size=4, opacity=0.5)
            st.plotly_chart(fig3)
        elif graph_selectbox == 'tGravityAccCorrelationYZ':
            fig4 = px.scatter_3d(df, x='subject', y='timepoint', z='tGravityAccCorrelationYZ',
                                color='Activity', height=700, width=800)
            fig4.update_traces(mode='markers', marker_size=4, opacity=0.5)
            st.plotly_chart(fig4)
        elif graph_selectbox == 'angletBodyGyroMeanGravityMean':
            fig5 = px.scatter_3d(df, x='subject', y='timepoint', z='angletBodyGyroMeanGravityMean',
                                color='Activity', height=700, width=800)
            fig5.update_traces(mode='markers', marker_size=4, opacity=0.5)
            st.plotly_chart(fig5)


    if add_selectbox == 'Batch':
        st.title("Batch Movement Prediction")

#       import io

        st.set_option('deprecation.showfileUploaderEncoding', False)

        file_buffer = st.file_uploader("Upload csv file for predictions", type=["csv"])

        if file_buffer is None:
            st.markdown('Upload a file to use the batch prediction feature')
        elif file_buffer is not None:
            data = pd.read_csv(file_buffer)
            predictions = predict_model(estimator=model,data=data)
            st.write(predictions)
#        text_io = io.TextIOWrapper(file_buffer)
if __name__ == '__main__':
    run()

