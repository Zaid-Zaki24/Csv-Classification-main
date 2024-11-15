import streamlit as st
import pandas as pd
import plotly.express as px


def plot_graph(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    st.header("Data Visualization")

    category_options = ['A', 'B', 'C']
    selected_categories = st.multiselect(
        "Select Categories", category_options, default=category_options)

    if selected_categories:
        filtered_df = df[df['Category'].isin(selected_categories)]

        fig_line = px.line(filtered_df, x='Date', y='Value',
                           color='Category', title='Value over Time')
        st.plotly_chart(fig_line)

        fig_bar = px.bar(filtered_df, x='Category', y='Value',
                         color='Category', title='Value by Category')
        st.plotly_chart(fig_bar)
