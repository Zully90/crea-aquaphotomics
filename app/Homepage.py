import streamlit as st
import numpy as np
import pandas as pd
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from sklearn.pipeline import Pipeline
from sklearn.neighbors import LocalOutlierFactor

from helpers.src.pca import PCA

import plotly.express as px
import plotly.graph_objects as go

import warnings
warnings.filterwarnings("ignore")

from helpers.src.preprocessing import msc_processing, derivate_processing, norm_preprocessing

from scipy import stats

# Titolo
st.title('Aquaphotomics')
st.text("Pagina di esempio per visualizzare i dati di uno studio di Aquaphotomics")

# Data import
raw = pd.read_excel("./data.xlsx")

raw.rename(
    columns={raw.columns[0]: "ID", raw.columns[1]: "Timestamp"},
    inplace = True
)

st.header("Dati Grezzi")
st.dataframe(raw)

### ---- SIDEBAR ---- ###
st.sidebar.header('Pipeline Steps')
use_smoothing = st.sidebar.checkbox('Activate Smoothing')
use_II_derivative = st.sidebar.checkbox('Activate Second Derivative')
use_msc = st.sidebar.checkbox('Activate MSC')
use_normalization = st.sidebar.checkbox('Use Normalization')

# Define the pipeline steps based on user selections
pipeline_steps = []

if use_smoothing:
    pipeline_steps.append(('Smoothing', derivate_processing(window_length=5, polyorder=1, derivate_order= 0)))
if use_II_derivative:
    pipeline_steps.append(('Derivative', derivate_processing(window_length=21, polyorder=2, derivate_order= 2)))
if use_msc:
    pipeline_steps.append(('MSC', msc_processing()))
if use_normalization:
    pipeline_steps.append(('Normalization', norm_preprocessing()))

# Create the scikit-learn pipeline
pipeline = Pipeline(pipeline_steps)

spectra = raw.loc[:, 908.1:1676.2].set_index(raw.ID)

### ---- OUTLIERS ---- ###


def lof_outlier_detection(data, contamination=0.1):
    lof = LocalOutlierFactor(contamination=contamination)
    labels = lof.fit_predict(data)
    outliers = data[labels == -1]
    return outliers

# Sidebar checkbox to activate outlier detection
st.sidebar.header('Outlier check')
enable_outlier_detection = st.sidebar.checkbox("Enable Outlier Detection")

if enable_outlier_detection:
    st.subheader("Outlier Detection Activated")

    # Perform outlier detection
    outliers = lof_outlier_detection(spectra)

    # Display the outliers
    # st.write("Outliers:")
    # st.write(outliers.index)
else:
    st.write("Outlier detection is disabled.")
    
out_index = outliers.index
spectra.drop(out_index, inplace=True)
raw = raw[~raw['Timestamp'].isin(out_index)]


### ----- SPECTRA PLOTS ----- ###

if pipeline_steps != []:
    # Full spectrum plot
    spectra_prepro = pd.DataFrame(pipeline.transform(spectra), index=spectra.index, columns=spectra.columns)
    spectra_prepro_t = spectra_prepro.transpose().reset_index()
    fig = px.line(spectra_prepro_t, x='index', y=spectra_prepro_t.columns[1:])
    
    # Water band    
    spectra_prepro_water = pd.DataFrame(pipeline.transform(spectra), index=spectra.index, columns=spectra.columns).loc[:, 1298.344:1601.868]
    spectra_prepro_water_t = spectra_prepro_water.transpose().reset_index()
    fig2 = px.line(spectra_prepro_water_t, x='index', y=spectra_prepro_water_t.columns[1:])
    
    # Loadings plot
    st.sidebar.header('Loadings Plot')
    modello = PCA(values=spectra_prepro.values, components=5)
    loadings = pd.DataFrame(modello.loadings, index=spectra_prepro.columns, columns=[f"comp_{i+1}" for i in range(modello.components)])
    selected_columns = st.sidebar.multiselect('Select the loading you whish to see', loadings.columns)
    loadings_plot = px.line(loadings[selected_columns])

else:
    # Full spectrum plot
    spectra_t = spectra.transpose().reset_index()
    fig = px.line(spectra_t, x='index', y=spectra_t.columns[1:])
    
    # Water band    
    spectra_t_water = spectra.loc[:, 1298.344:1601.868].transpose().reset_index()
    fig2 = px.line(spectra_t_water, x='index', y=spectra_t_water.columns[1:])
    
    # Loadings plot
    st.sidebar.header('Loadings Plot')
    modello = PCA(values=spectra.values, components=5)
    loadings = pd.DataFrame(modello.loadings, index=spectra.columns, columns=[f"comp_{i+1}" for i in range(modello.components)])
    selected_columns = st.sidebar.multiselect('Select the loading you whish to see', loadings.columns)
    loadings_plot = px.line(loadings[selected_columns])
    
    spectra_prepro = None
    
x_tickvals = [i for i in range(125)] 
x_ticktext = spectra.columns

fig.update_layout(
    xaxis_title='Wavelengths',
    yaxis_title='Absorbance',
    legend_title='Sample Names',
    height=600,
    width=1000,
    #title='Full Spectrum'
)

fig2.update_layout(
    xaxis_title='Wavelengths',
    yaxis_title='Absorbance',
    legend_title='Sample Names',
    height=600,
    width=800,
    #title='Water Spectrum'
)

fig.update_xaxes(ticktext=x_ticktext, tickvals=x_tickvals)
fig.update_traces(hovertemplate='Wavelength: %{x}<br>Absorbance: %{y}')

# # Show the plot
st.header("Full spectra plot")
st.plotly_chart(fig, theme=None, use_container_width=False,)

st.header("WaterBand plot")
st.plotly_chart(fig2, theme=None, use_container_width=False,)

st.header("Spectra-Like Loadings plot")
st.plotly_chart(loadings_plot)

### ---- AQUAGRAMS ----- ###

st.header("Aquagram")

if spectra_prepro is not None:
    aqua_df = spectra_prepro.copy()
else:
    aqua_df = raw.copy()

selected_wl = [1341.705, 1366.482, 1372.677, 1409.843, 1428.426, 1440.814, 1453.203, 1459.398, 1477.981, 1490.369, 1515.147]
aquagram_df = aqua_df.loc[:, selected_wl]

aquagram_df.insert(
    loc=0,
    column="Timestamp",
    value=raw['Timestamp'].values
)

# Convert the 'timestamp' column to datetime
aquagram_df.set_index('Timestamp', inplace=True)
# Resample by hour
hourly_aquagram = aquagram_df.resample('H').mean()

st.header("Data resampled by hour")
st.dataframe(hourly_aquagram)

# Set the column names as the categories for the radar plot
categories = hourly_aquagram.columns

converted_categories = []
for item in categories:
    converted_item = str(np.round(item, 0))+" nm"  # Conve
    converted_categories.append(converted_item)


# Create traces for each line in the radar plot
traces = []
for i, row in hourly_aquagram.iterrows():
    values = row.values.tolist()
    #values.append(values[0])  # Add the first value at the end to close the radar plot
    trace = go.Scatterpolar(
        r=values,
        theta=converted_categories,
        fill='toself',
        name=f'{i}'
    )
    traces.append(trace)
    
# Create the layout for the radar plot
layout = go.Layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[hourly_aquagram.min().min(), hourly_aquagram.max().max()]  # Adjust the range based on your data
        )
    ),
    showlegend=True,
    # width=1000,  # Set the width of the plot
    # height=800  # Set the height of the plot
)

st.header("Plot")

# Create the figure
aquagramma = go.Figure(data=traces, layout=layout)

# Show the plot
st.plotly_chart(aquagramma, theme=None, use_container_width=False,)



# Inserire piccolo controllo outliers (checkbox)
# Inserire Sfondo (?)
# Plot Spettro intero - Mancano Nomi sugli assi e dei campioni - DONE
# Plot banda 1300-1600 - DONE
# SpiderPlot delle 12 WL - DONE
# Plot PCA con scelta di quale componente mettere su assi (sia scatter che loadings spectra-like) - DONE
