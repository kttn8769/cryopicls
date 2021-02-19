"""Web app for visualizing cryoPICLS results."""

import os
import argparse

import numpy as np
import pandas as pd
import plotly.express as px
import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.dependencies import Input, Output, State
from dash_table import DataTable


templates = [
    {'label': x, 'value': x}
    for x in
    ['ggplot2', 'seaborn', 'simple_white', 'plotly', 'plotly_white', 'plotly_dark',
     'presentation', 'xgridoff', 'ygridoff', 'gridon', 'none']]

df = pd.DataFrame()
datatable_data = dict()
datatable_columns = [
    dict(id='datatable_groups', name='Groups'),
    dict(id='datatable_num_samples', name='Num Samples', type='numeric')
]
df_mins = pd.Series(dtype='float64')
df_maxs = pd.Series(dtype='float64')
clustering_result_file = None
projection_result_file = None

# dash.Dash automatically loads .css files in the assets directory.
app = dash.Dash(__name__, title="cryoPICLS", suppress_callback_exceptions=True)


navvar = dbc.Row([
    dbc.NavbarSimple(
        brand='cryoPICLS Visualizer',
        brand_style={'color': 'white', 'font-weight': 'bold', 'font-size': '2rem'}, color='primary',
        fluid=True,
        style={'min-width': '100vw'},
        className='pl-0')
])


def get_options():
    axes = df.drop('cluster', axis=1, errors='ignore').columns
    options = [{'label': x, 'value': x} for x in axes]
    return options


def get_color_switch_disable():
    if 'cluster' in df.columns:
        val = False
    else:
        val = True
    return val


def get_color(color_by_cluster):
    if color_by_cluster:
        color = 'cluster'
    else:
        color = None
    return color


def get_range(axis, margin=0.5):
    vmin = df_mins[axis] - margin
    vmax = df_maxs[axis] + margin
    return [vmin, vmax]


def create_datatable_data(df_in):
    groups = []
    num_samples = []
    if 'cluster' in df_in.columns:
        cluster_ids, cluster_num_samples = np.unique(df_in['cluster'], return_counts=True)
        cluster_ids = [f'cluter_{x}' for x in cluster_ids]
        cluster_num_samples = list(cluster_num_samples)
        groups += cluster_ids
        num_samples += cluster_num_samples
    groups.append('Total')
    num_samples.append(df_in.shape[0])

    data = [
        {'datatable_groups': x, 'datatable_num_samples': y}
        for x, y in zip(groups, num_samples)
    ]

    return data


def get_file_info():
    info = []
    if clustering_result_file:
        info.append(html.H6('Clustering result:', style={'font-weight': 'bold'}))
        info.append(html.Div(clustering_result_file, className='ml-4'))
    if projection_result_file:
        info.append(html.H6('Projection result:', style={'font-weight': 'bold'}, className='mt-2'))
        info.append(html.Div(projection_result_file, className='ml-4'))
    return info


def create_container_scatter_3d():
    options = get_options()

    container_scatter_3d = dbc.Container([
        navvar,

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Scatter 3D', className='card-title'),
                        html.H6('X-axis:'),
                        dcc.Dropdown(
                            id='container-scatter-3d-dropdown-x',
                            options=options,
                            value=options[0]['value']
                        ),
                        html.H6('Y-axis:'),
                        dcc.Dropdown(
                            id='container-scatter-3d-dropdown-y',
                            options=options,
                            value=options[1]['value']
                        ),
                        html.H6('Z-axis:'),
                        dcc.Dropdown(
                            id='container-scatter-3d-dropdown-z',
                            options=options,
                            value=options[2]['value']
                        ),
                        html.H6('Plot theme:'),
                        dcc.Dropdown(
                            id='container-scatter-3d-dropdown-theme',
                            options=templates,
                            value='plotly_white'
                        ),
                        html.H6('Marker size:'),
                        dcc.Slider(
                            id='container-scatter-3d-slider-marker',
                            min=1, max=10,
                            step=1,
                            value=3,
                            marks={
                                1: '1',
                                10: '10'
                            },
                            dots=True
                        ),
                        html.H6('Color by cluster:'),
                        daq.BooleanSwitch(
                            id='container-scatter-3d-switch-color',
                            on=not get_color_switch_disable(),
                            disabled=get_color_switch_disable()
                        ),
                        dbc.Button(
                            'Update', id='container-scatter-3d-card-1-button-update',
                            outline=True, color='primary', n_clicks=0
                        )
                    ])
                ], id='container-scatter-3d-card-1', className='mt-3'),
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Data statistics', className='card-title'),
                        DataTable(
                            columns=datatable_columns,
                            data=datatable_data,
                            cell_selectable=False
                        )
                    ])
                ], id='container-scatter-3d-card-2', className='mt-3')
            ], width={'size': 4}, style={'min-height': '100vh', 'background-color': '#f5f5f5'}),

            dbc.Col([
                dcc.Graph(id='container-scatter-3d-graph-1', figure={}, style={'height': '70vh', 'display': 'none'}),
                dbc.Alert(id='container-scatter-3d-text', color='light')
            ], width={'size': 8}, style={'min-height': '100vh'}),
        ])
    ], fluid=True)

    return container_scatter_3d


@app.callback(
    [Output('container-scatter-3d-graph-1', 'figure'),
     Output('container-scatter-3d-graph-1', 'style'),
     Output('container-scatter-3d-text', 'children')],
    Input('container-scatter-3d-card-1-button-update', 'n_clicks'),
    [State('container-scatter-3d-graph-1', 'style'),
     State('container-scatter-3d-dropdown-x', 'value'),
     State('container-scatter-3d-dropdown-y', 'value'),
     State('container-scatter-3d-dropdown-z', 'value'),
     State('container-scatter-3d-switch-color', 'on'),
     State('container-scatter-3d-slider-marker', 'value'),
     State('container-scatter-3d-dropdown-theme', 'value')]
)
def update_scatter3d(n_clicks, style, x_axis, y_axis, z_axis, color_by_cluster,
                     marker_size, theme):
    color = get_color(color_by_cluster)

    fig = px.scatter_3d(
        data_frame=df,
        x=x_axis,
        y=y_axis,
        z=z_axis,
        color=color,
        template=theme,
        range_x=get_range(x_axis),
        range_y=get_range(y_axis),
        range_z=get_range(z_axis)
    )

    fig.update_traces(
        marker_size=marker_size
    )

    if color == 'cluster':
        fig.update_layout(
            legend_title_text='ClusterID',
        )

    style['display'] = 'block'

    text = get_file_info()

    return fig, style, text


def create_container_scatter_2d():
    options = get_options()

    container_scatter_2d = dbc.Container([
        navvar,

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Scatter 2D', className='card-title'),
                        html.H6('X-axis:'),
                        dcc.Dropdown(
                            id='container-scatter-2d-dropdown-x',
                            options=options,
                            value=options[0]['value']
                        ),
                        html.H6('Y-axis:'),
                        dcc.Dropdown(
                            id='container-scatter-2d-dropdown-y',
                            options=options,
                            value=options[1]['value']
                        ),
                        html.H6('Plot theme:'),
                        dcc.Dropdown(
                            id='container-scatter-2d-dropdown-theme',
                            options=templates,
                            value='plotly_white'
                        ),
                        html.H6('Marker size:'),
                        dcc.Slider(
                            id='container-scatter-2d-slider-marker',
                            min=1, max=10,
                            step=1,
                            value=3,
                            marks={
                                1: '1',
                                10: '10'
                            },
                            dots=True
                        ),
                        html.H6('Color by cluster:'),
                        daq.BooleanSwitch(
                            id='container-scatter-2d-switch-color',
                            on=not get_color_switch_disable(),
                            disabled=get_color_switch_disable()
                        ),
                        dbc.Button(
                            'Update', id='container-scatter-2d-card-1-button-update',
                            outline=True, color='primary', n_clicks=0
                        )
                    ])
                ], id='container-scatter-2d-card-1', className='mt-3'),
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Data statistics', className='card-title'),
                        DataTable(
                            columns=datatable_columns,
                            data=datatable_data,
                            cell_selectable=False
                        )
                    ])
                ], id='container-scatter-2d-card-2', className='mt-3')
            ], width={'size': 4}, style={'min-height': '100vh', 'background-color': '#f5f5f5'}),

            dbc.Col([
                dcc.Graph(id='container-scatter-2d-graph-1', figure={}, style={'height': '70vh', 'display': 'none'}),
                dbc.Alert(id='container-scatter-2d-text', color='light')
            ], width={'size': 8}, style={'min-height': '100vh'}),
        ])
    ], fluid=True)

    return container_scatter_2d


@app.callback(
    [Output('container-scatter-2d-graph-1', 'figure'),
     Output('container-scatter-2d-graph-1', 'style'),
     Output('container-scatter-2d-text', 'children')],
    Input('container-scatter-2d-card-1-button-update', 'n_clicks'),
    [State('container-scatter-2d-graph-1', 'style'),
     State('container-scatter-2d-dropdown-x', 'value'),
     State('container-scatter-2d-dropdown-y', 'value'),
     State('container-scatter-2d-switch-color', 'on'),
     State('container-scatter-2d-slider-marker', 'value'),
     State('container-scatter-2d-dropdown-theme', 'value')],
)
def update_scatter2d(n_clicks, style, x_axis, y_axis, color_by_cluster,
                     marker_size, theme):
    color = get_color(color_by_cluster)

    fig = px.scatter(
        data_frame=df,
        x=x_axis,
        y=y_axis,
        color=color,
        template=theme,
        range_x=get_range(x_axis),
        range_y=get_range(y_axis)
    )

    fig.update_traces(
        marker_size=marker_size
    )

    if color == 'cluster':
        fig.update_layout(
            legend_title_text='ClusterID'
        )

    style['display'] = 'block'

    text = get_file_info()

    return fig, style, text


def create_container_hist_1d():
    options = get_options()

    container_hist_1d = dbc.Container([
        navvar,

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Histgram 1D', className='card-title'),
                        html.H6('X-axis:'),
                        dcc.Dropdown(
                            id='container-hist-1d-dropdown-x',
                            options=options,
                            value=options[0]['value']
                        ),
                        html.H6('Plot theme:'),
                        dcc.Dropdown(
                            id='container-hist-1d-dropdown-theme',
                            options=templates,
                            value='plotly_white'
                        ),
                        html.H6('Color by cluster:'),
                        daq.BooleanSwitch(
                            id='container-hist-1d-switch-color',
                            on=not get_color_switch_disable(),
                            disabled=get_color_switch_disable()
                        ),
                        dbc.Button(
                            'Update', id='container-hist-1d-card-1-button-update',
                            outline=True, color='primary', n_clicks=0
                        )
                    ])
                ], id='container-hist-1d-card-1', className='mt-3'),
                dbc.Card([
                    dbc.CardBody([
                        html.H5('Data statistics', className='card-title'),
                        DataTable(
                            columns=datatable_columns,
                            data=datatable_data,
                            cell_selectable=False
                        )
                    ])
                ], id='container-hist-1d-card-2', className='mt-3')
            ], width={'size': 4}, style={'min-height': '100vh', 'background-color': '#f5f5f5'}),

            dbc.Col([
                dcc.Graph(id='container-hist-1d-graph-1', figure={}, style={'height': '70vh', 'display': 'none'}),
                dbc.Alert(id='container-hist-1d-text', color='light'),
            ], width={'size': 8}, style={'min-height': '100vh'}),
        ])
    ], fluid=True)

    return container_hist_1d


@app.callback(
    [Output('container-hist-1d-graph-1', 'figure'),
     Output('container-hist-1d-graph-1', 'style'),
     Output('container-hist-1d-text', 'children')],
    Input('container-hist-1d-card-1-button-update', 'n_clicks'),
    [State('container-hist-1d-graph-1', 'style'), State('container-hist-1d-dropdown-x', 'value'),
     State('container-hist-1d-switch-color', 'on'),
     State('container-hist-1d-dropdown-theme', 'value')],
)
def update_hist1d(n_clicks, style, x_axis, color_by_cluster, theme):
    color = get_color(color_by_cluster)

    fig = px.histogram(
        data_frame=df,
        x=x_axis,
        color=color,
        marginal='rug',
        opacity=0.7,
        template=theme,
        barmode='overlay',
        range_x=get_range(x_axis)
    )

    if color == 'cluster':
        fig.update_layout(
            legend_title_text='ClusterID'
        )

    style['display'] = 'block'

    text = get_file_info()

    return fig, style, text


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__
    )
    parser.add_argument('--port', default=8050, type=int, help='Port number.')
    parser.add_argument('--debug', action='store_true', help='Run app in debug mode.')
    parser.add_argument('--clustering-result', type=str, help='Clustering result file of cryoPICLS.')
    parser.add_argument('--projection-result', type=str, help='Projection result file of cryoPICLS.')
    parser.add_argument('--scatter2d', action='store_true', help='2D scatter plot.')
    parser.add_argument('--scatter3d', action='store_true', help='3D scatter plot.')
    parser.add_argument('--hist1d', action='store_true', help='1D histogram plot.')
    parser.add_argument('--stride', type=int, default=1, help='Only use one in every --stride number of samples, to reduce computational load for a large dataset.')

    args = parser.parse_args()

    assert args.scatter2d + args.scatter3d + args.hist1d == 1, 'Must specify either one of --scatter2d, --scatter3d, --hist1d.'
    assert args.stride > 0, '--stride must be a positive integer number.'

    return args


def main():
    global df, datatable_data, df_mins, df_maxs, clustering_result_file, projection_result_file

    args = parse_args()

    df_clustering = pd.DataFrame()
    df_projection = pd.DataFrame()

    if args.clustering_result:
        assert os.path.exists(args.clustering_result), f'--clustering-result {args.clustering_result} : File not found.'
        df_clustering = pd.read_pickle(args.clustering_result)
        clustering_result_file = args.clustering_result

    if args.projection_result:
        assert os.path.exists(args.projection_result), f'--projection-result {args.projection_result} : File not found.'
        df_projection = pd.read_pickle(args.projection_result)
        projection_result_file = args.projection_result

    if (not df_clustering.empty) and (not df_projection.empty):
        assert df_clustering.shape[0] == df_projection.shape[0], f'Mismatch in the nubmer of samples. clustering: {df_clustering.shape[0]}, projection: {df_projection.shape[0]}'
        # Assign cluster labels to projections, and use them for plotting.
        df = df_projection.join(df_clustering['cluster'])
    elif not df_clustering.empty:
        df = df_clustering
    else:
        df = df_projection

    datatable_data = create_datatable_data(df)

    if args.stride > 1:
        df = df[::args.stride]

    if 'cluster' in df.columns:
        df.sort_values(by='cluster', axis=0, inplace=True)
        df['cluster'] = df['cluster'].apply(lambda x: f'cluster_{x}')

    df_mins = df.min()
    df_maxs = df.max()

    n_samples, n_dims = df.drop('cluster', axis=1, errors='ignore').shape

    if args.scatter3d:
        assert n_dims >= 3, f'Data dimension is {n_dims}, which cannot be used for 3D scatter plotting.'
        app.layout = create_container_scatter_3d()
    elif args.scatter2d:
        assert n_dims >= 2, f'Data dimension is {n_dims}, which cannot be used for 2D scatter plotting.'
        app.layout = create_container_scatter_2d()
    elif args.hist1d:
        app.layout = create_container_hist_1d()

    app.run_server(debug=args.debug, port=args.port)


if __name__ == "__main__":
    main()
