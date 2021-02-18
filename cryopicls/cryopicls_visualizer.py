"""Web app for visualizing cryoPICLS results."""

import pickle
import argparse
import numpy as np
import pandas as pd

import plotly.express as px
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

df = None

# dash.Dash automatically loads .css files in the assets directory.
# app = dash.Dash(__name__, title="cryoPICLS")
app = dash.Dash(__name__, title='cryoPICLS', external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    html.Div(id='data-store-1', style={'display': 'none'}),

    dbc.Row([
        dbc.NavbarSimple(
            brand='cryoPICLS Visualizer',
            brand_style={'color': 'white', 'font-weight': 'bold', 'font-size': '2rem'}, color='primary',
            fluid=True,
            style={'min-width': '100vw'},
            className='pl-0')
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5('Inputs', className='card-title'),
                    html.P('cryoPICLS file'),
                    dbc.Input(
                        placeholder='Specify path',
                        id='card-1-cryopicls-result-file'
                    ),
                    dbc.Button(
                        'Load', id='card-1-button-load',
                        outline=True, color='primary', n_clicks=0
                    )
                ])
            ], id='card-1', className='mt-3'),

        #     dbc.Card([
        #         dbc.CardBody([
        #             html.H5('Clustering', className='card-title'),
        #             html.P('Algorithm:'),
        #             dcc.Dropdown(
        #                 id='card-2-algorithm',
        #                 options=[
        #                     {'label': 'K-Means', 'value': 'k-means'},
        #                     {'label': 'Auto-GMM', 'value': 'auto-gmm'},
        #                     {'label': 'X-Means', 'value': 'x-means'},
        #                     {'label': 'G-Means', 'value': 'g-means'}
        #                 ],
        #                 placeholder='Select clustering algorithm'
        #             ),
        #             html.Div(id='card-2-content', children=[])
        #         ])
        #     ], id='card-2', className='mt-3'),

        #     dbc.Card([
        #         dbc.CardBody([
        #             html.H5('Dimensionaliry Reduction', className='card-title'),
        #             html.P('Algorithm:'),
        #             dcc.Dropdown(
        #                 id='card-3-algorithm',
        #                 options=[
        #                     {'label': 'PCA', 'value': 'pca'},
        #                     {'label': 'UMAP', 'value': 'umap'}
        #                 ],
        #                 placeholder='Select algorithm'
        #             ),
        #             html.Div(id='card-3-content', children=[])
        #         ])
        #     ], id='card-3', className='mt-3')
        ], width={'size': 4}, style={'min-height': '100vh', 'background-color': '#f5f5f5'}),

        dbc.Col([
            dcc.Graph(id='scatter3d', figure={}, style={'height': '70vh', 'display': 'none'}),
        ], width={'size': 8}, style={'min-height': '100vh'}),
    ])

], fluid=True)


@app.callback(
    [Output('scatter3d', 'figure'), Output('scatter3d', 'style')],
    Input('data-store-1', 'children'),
    [State('scatter3d', 'style')],
)
def update_scatter3d(data, style):
    df = pd.read_json(data)

    dff = df.sort_values(by='cluster', axis=0)
    dff['cluster'] = dff['cluster'].apply(lambda x: f'cluster_{x}')

    fig = px.scatter_3d(
        data_frame=dff,
        x='dim_0',
        y='dim_1',
        z='dim_2',
        color='cluster',
        # labels={'cluster': 'cluster'},
        # opacity=0.7,
        template='plotly',
        # symbol='cluster',
        # text='cluster'
    )

    fig.update_traces(
        marker_size=3
    )

    fig.update_layout(
        legend_title_text='ClusterID'
    )

    style['display'] = 'block'

    return fig, style


@app.callback(
    Output('data-store-1', 'children'),
    Input('card-1-button-load', 'n_clicks'),
    State('card-1-cryopicls-result-file', 'value')
)
def load_inputs(n_clicks, cryopicls_result_file):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    try:
        df = pd.read_pickle(cryopicls_result_file)
    except FileNotFoundError:
        raise dash.exceptions.PreventUpdate

    return df.to_json()


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument('--port', default=8050, type=int, help='Port number.')
    parser.add_argument('--debug', action='store_true', help='Run app in debug mode.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # app.run_server(debug=args.debug, port=args.port)
    app.run_server(debug=True, port=args.port)


if __name__ == "__main__":
    main()
