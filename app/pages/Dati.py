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
st.title('FANTACITORIANAL AWAAAEEEEE')

#Opening the worksheet by using Worksheet ID
workbook = gc.open_by_key("1FAx14D7s9jcuakh8iyrgD8bmf7sKdeH6On5ppL5cX3s") #Selecting which sheet to pulling the data
sheet = workbook.worksheet('Dati') #Pulling the data and transform it to the data frame
values = sheet.get_all_values()
data = pd.DataFrame(values[1:], columns = values[0])

conteggio_partiti = pd.DataFrame(data={
    "Rassamblement": data.Partito.value_counts().index,
    "Numero di membri": data.Partito.value_counts(),
    }   
).reset_index(drop=True)

AgGrid(data, fit_columns_on_grid_load=True)

fig = px.pie(
    conteggio_partiti,
    values='Numero di membri',
    names='Rassamblement',
    title='Distribuzione dei Rassamblement'
)

st.plotly_chart(fig, use_container_width=True)
