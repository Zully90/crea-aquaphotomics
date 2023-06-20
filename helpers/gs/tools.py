# Parametri Grid Search

def parameter_generator():
    from helpers.qpowermetal import pipeline_transformations as qpipeline
    import itertools
    from sklearn.preprocessing import StandardScaler
    
    smooth_comb = [
        {
            "window_length": window_length,
            "polyorder": polyorder,
            "derivate_order": 0,
        } 
        for window_length, polyorder in itertools.product(
            range(3, 25), 
            range(0, 5)) if window_length % 2 != 0 and polyorder < window_length]

    derivate_comb = [
        {
            "window_length": window_length,
            "polyorder": polyorder,
            "derivate_order": derivate_order,
        } 
        for window_length, polyorder, derivate_order in itertools.product(
            range(3, 25),
            range(1, 5),
            range(1, 4)) if window_length % 2 != 0 and polyorder < window_length and polyorder >= derivate_order]

    # Dichiaro i parametri
    
    mean_centering = StandardScaler(with_mean=True, with_std=False)
    
    parameters = {
        # "Log10": [None, np.log10],
        "scaler": [None, mean_centering],
        "smoothing__kw_args": smooth_comb,
        "derivate__kw_args": derivate_comb,
        "snv": [None, qpipeline.snv_processing()],
        "model__n_components": [4, 6, 8, 12]
    }
    
    return smooth_comb, derivate_comb, parameters

def model_generator(spectra, target, parameters):
    from sklearn.pipeline import Pipeline
    from helpers.qpowermetal import pipeline_transformations as qpipeline
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.preprocessing import StandardScaler

    # Pipeline
    preprocessing = Pipeline(steps=[
        # ("Log10", "passthrough"),
        ("scale", "passthrough"),
        ("smoothing", qpipeline.derivate_processing(window_length=11, polyorder=1, derivate_order=0)),
        ("derivate", qpipeline.derivate_processing(window_length=11, polyorder=3, derivate_order=2)),
        ("snv", "passthrough"),
        ("model", PLSRegression(
            n_components=6,
            scale=False,
            max_iter=500,
        ))
    ])

    # Effettuo la GS

    if spectra.shape[0] > 10:
        search = RandomizedSearchCV(
            preprocessing,
            parameters,
            n_iter=50,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
            cv=10, # SMARCARE QUESTO.
        )

    else:
        search = RandomizedSearchCV(
            preprocessing,
            parameters,
            n_iter=50,
            n_jobs=-1,
            scoring="neg_root_mean_squared_error",
            refit="r2",
            cv=2
        )
        
    search.fit(spectra, target)
    
    return search

def model_generator_halving(spectra, target, parameters):
    from sklearn.pipeline import Pipeline
    from helpers.qpowermetal import pipeline_transformations as qpipeline
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.experimental import enable_halving_search_cv  # noqa
    from sklearn.model_selection import HalvingRandomSearchCV
    from sklearn.preprocessing import StandardScaler
    
    mean_centering = StandardScaler(with_mean=True, with_std=False)

    # Pipeline
    preprocessing = Pipeline(steps=[
        # ("Log10", "passthrough"),
        ("scaler", mean_centering),
        ("smoothing", qpipeline.derivate_processing(window_length=11, polyorder=1, derivate_order=0)),
        ("derivate", qpipeline.derivate_processing(window_length=11, polyorder=3, derivate_order=2)),
        ("snv", "passthrough"),
        ("model", PLSRegression(
            n_components=6,
            scale=False,
            max_iter=500,
        ))
    ])

    # Effettuo la GS

    if spectra.shape[0] > 10:
        search = HalvingRandomSearchCV(
            preprocessing,
            parameters,
            # resource='n_estimators',
            # max_resources=10,
            random_state=0,
            aggressive_elimination=True,
            scoring="neg_root_mean_squared_error",
            cv=10, # SMARCARE QUESTO.
        )

    else:
        search = HalvingRandomSearchCV(
            preprocessing,
            parameters,
            # resource='n_estimators',
            # max_resources=10,
            random_state=0,
            aggressive_elimination=True,
            scoring="neg_root_mean_squared_error",
            cv=2
        )
        
    search.fit(spectra, target)
    
    return search

def reg_plot(results, best_model):
    
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from sklearn.metrics import r2_score as R2

    # Data
    bisector_data = np.linspace(min(results.experimental)-1, max(results.experimental)+1)

    # Some calculations
    r_quadro = R2(results.experimental, results.predicted).round(2)
    rmse = abs(best_model.best_score_.round(2))
    
    # Plots
    bisector = px.line(
        x = bisector_data,
        y = bisector_data,
    )

    bisector.update_traces(line_color='red', line_width=1)

    pred_exp = px.scatter(results, x="experimental", y="predicted", trendline="ols")

    fig = go.Figure(
        data = bisector.data + pred_exp.data,
        layout=go.Layout(
            annotations=[
                go.layout.Annotation(
                text=f"R2: {r_quadro}, RMSE: {rmse}",
                align='center',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.05,
                y=0.9,
                bordercolor='black',
                borderwidth=1
                )
            ]
        )
    )

    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
    )
    
    return fig