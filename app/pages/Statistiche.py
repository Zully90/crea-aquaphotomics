import gspread
from df2gspread import df2gspread as d2g
from oauth2client.service_account import ServiceAccountCredentials
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

#The scope is always look like this so we did not need to change anything
scope = [
    'https://spreadsheets.google.com/feeds',
    'https://www.googleapis.com/auth/drive'
] #Name of our Service Account Key
google_key_file = 'fantacitorio-bc6c1d886f45.json'
credentials = ServiceAccountCredentials.from_json_keyfile_name(google_key_file, scope)
gc = gspread.authorize(credentials)

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

st.set_page_config(
    page_title="Data Analysis with Q"
)

# Titolo
st.title('Statistiche varie')
st.header("Punti per partito")

# Opening the worksheet by using Worksheet ID
workbook = gc.open_by_key("1FAx14D7s9jcuakh8iyrgD8bmf7sKdeH6On5ppL5cX3s") #Selecting which sheet to pulling the data
sheet = workbook.worksheet('Punti per giornata') #Pulling the data and transform it to the data frame
values = sheet.get_all_values()
data = pd.DataFrame(values[1:], columns = values[0])

# Conversion to numeric columns
cols_to_convert = [col for col in data.iloc[:, 3:]]
for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    
# Punti per partito    
grouped = pd.DataFrame(data.groupby("Partito").agg("sum").sum(axis=1)).reset_index()
grouped.rename({0: "Punteggio"}, inplace=True, axis=1)

grouped

fig = px.pie(
    grouped,
    values='Punteggio',
    names='Partito',
    # title='Distribuzione dei Rassamblement'
)

st.plotly_chart(fig, use_container_width=True)