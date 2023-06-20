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
st.title('Risultati')
st.header("Punti per giornata")

# Opening the worksheet by using Worksheet ID
workbook = gc.open_by_key("1FAx14D7s9jcuakh8iyrgD8bmf7sKdeH6On5ppL5cX3s") #Selecting which sheet to pulling the data
sheet = workbook.worksheet('Punti per giornata') #Pulling the data and transform it to the data frame
values = sheet.get_all_values()
data = pd.DataFrame(values[1:], columns = values[0])

### Preparazione ###

# Conversion to numeric columns
cols_to_convert = [col for col in data.iloc[:, 3:]]
for col in cols_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    
AgGrid(data, fit_columns_on_grid_load=True)

melted = pd.melt(
    data,
    id_vars=['Cognome', "Team"],
    value_vars= data.columns[3:],
    value_name="Punteggio",
    var_name="Giornata"
)

melted_no_na = melted[melted.Team != "NA"]

st.header("Putei's trend")
fig = px.line(melted_no_na, x="Giornata", y="Punteggio", color="Cognome")
st.plotly_chart(fig, use_container_width=True)

### Classifica ###

data_notna = data[data.Team != "NA"]

classifica = {}
for team in data_notna.Team.unique():
    classifica[team] = data_notna[data_notna.Team == team].iloc[:, 3:].sum().sum().sum()

st.header("Classifica")
classifica = pd.DataFrame(classifica, index=["Punteggio"]).T.reset_index().rename({"index": "Team"}, axis=1).sort_values("Punteggio", ascending=False)
st.dataframe(classifica, use_container_width=True)

fig = px.bar(classifica, x='Team', y='Punteggio')
st.plotly_chart(fig, use_container_width=True)

### Top Players ###

st.header("Top Players")

lista_best = []
# Individuo i top di giornata
for giornata in data.iloc[:, 3:]:
    best_guy = data[data[giornata] == data[giornata].max()].loc[:, ["Cognome", "Team", giornata]].reset_index(drop=True)
    best_guy.rename(columns={giornata: "Punteggio"}, inplace=True)
    lista_best.append(best_guy)

top_players = pd.concat(lista_best).reset_index(drop=True)
top_players.index = np.arange(1, len(top_players) + 1)
st.dataframe(top_players, use_container_width=True)

fig = px.line(top_players, x="Cognome", y="Punteggio")
st.plotly_chart(fig, use_container_width=True)