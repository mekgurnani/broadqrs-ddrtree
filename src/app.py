from pathlib import Path
import json
import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import ecg_plot

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import pandas as pd
import dash
from dash import dcc, html, Input, Output
from flask_caching import Cache

# Initialize Dash app
app = dash.Dash(__name__)
server = app.server

# Configure caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Load static data once and cache it
@cache.memoize()
def load_static_data():
    model_info = json.load(open('MODEL_INFO.json'))
    i_factors = json.load(open('val_i_factors.json'))
    i_factor_ids = [f'{i + 1}' for i in i_factors['indices']]
    
    group_medians_3d = np.load('group28_medians_3d.npy')
    group_stds_3d = np.load('group28_stds_3d.npy')
    
    decoded_ecgs_array_12L = [group_medians_3d[i] for i in range(group_medians_3d.shape[0])]
    decoded_stds_array_12L = [group_stds_3d[i] for i in range(group_stds_3d.shape[0])]
    
    return model_info, i_factors, i_factor_ids, decoded_ecgs_array_12L, decoded_stds_array_12L

MODEL_INFO, i_factors, i_factor_ids, decoded_ecgs_array_12L, decoded_stds_array_12L = load_static_data()

# 12-LEAD ECG LEAD NAMES
ECG_12_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# COLORMAP TEMPLATES FOR HORIZONTAL AND VERTICAL LEADS
LEAD_COLORS_plotly = {
    'I': 'indigo',
    'II': 'slateblue',
    'III': 'teal',
    'aVR': 'limegreen',
    'aVL': 'midnightblue',
    'aVF': 'mediumturquoise',
    'V1': 'darkslateblue',
    'V2': 'mediumslateblue',
    'V3': 'mediumslateblue',
    'V4': 'mediumslateblue',
    'V5': 'mediumorchid',
    'V6': 'fuchsia'
}
LEAD_COLORS = defaultdict(lambda: 'black', LEAD_COLORS_plotly)

def plot_ecg_plotly(ecg_signal, sampling_rate, lead_names=None, subplot_shape=None, ylim=None, share_ylim=True,
                    title=None, std=None, percentiles=None, figsize=None, show_axes=True, show_legend=False, **kwargs):
    if len(ecg_signal.shape) != 2:
        raise ValueError('ECG signal must have shape: (num_samples, num_leads)')

    time_index = np.arange(ecg_signal.shape[0]) / sampling_rate
    num_leads = ecg_signal.shape[1]

    ylim_ = (np.min(ecg_signal), np.max(ecg_signal)) if ylim is None and share_ylim else ylim

    lead_names = lead_names or [f'Lead {i + 1}' for i in range(num_leads)]
    lead_colors = LEAD_COLORS_plotly if lead_names else dict(zip(lead_names, LEAD_COLORS_plotly))

    subplot_shape = subplot_shape or (num_leads, 1)
    subplot_height_cm, subplot_width_cm = 8, 6.2

    row_heights = [subplot_height_cm / (subplot_height_cm * subplot_shape[0])] * subplot_shape[0]
    column_widths = [subplot_width_cm / (subplot_width_cm * subplot_shape[1])] * subplot_shape[1]

    fig = make_subplots(
        rows=subplot_shape[0], cols=subplot_shape[1],
        subplot_titles=lead_names,
        row_heights=row_heights,
        column_widths=column_widths,
        vertical_spacing=0.08,
        shared_yaxes=share_ylim,
        x_title="Time (seconds)", y_title="Amplitude (mV)"
    )

    for i in range(num_leads):
        row, col = i % subplot_shape[0] + 1, i // subplot_shape[0] + 1
        fig.add_trace(go.Scatter(
            x=time_index, y=ecg_signal[:, i], mode='lines', name=lead_names[i],
            line=dict(color=lead_colors[lead_names[i]]), showlegend=show_legend
        ), row=row, col=col)
        
        fig.update_xaxes(dtick=0.2, row=row, col=col)
        fig.update_yaxes(dtick=0.5, row=row, col=col)

        if std is not None:
            fig.add_trace(go.Scatter(
                x=np.concatenate([time_index, time_index[::-1]]),
                y=np.concatenate([ecg_signal[:, i] - std[:, i], (ecg_signal[:, i] + std[:, i])[::-1]]),
                fill='toself', fillcolor=lead_colors[lead_names[i]], line=dict(color='rgba(255,255,255,0)'), opacity=0.2, showlegend=False
            ), row=row, col=col)

        if percentiles is not None:
            fig.add_trace(go.Scatter(
                x=time_index, y=percentiles[0][:, i], mode='lines', line=dict(width=0), showlegend=False
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=time_index, y=percentiles[1][:, i], fill='tonexty', mode='lines', line=dict(width=0), fillcolor=lead_colors[lead_names[i]], opacity=0.2, showlegend=False
            ), row=row, col=col)

    fig.update_layout(
        title=title, width=subplot_shape[1] * subplot_width_cm * 40, height=subplot_shape[0] * subplot_height_cm * 40
    )
    if ylim_ is not None:
        fig.update_yaxes(range=ylim_)
    if not show_axes:
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)

    return fig

# Read and process the tree data
broadqrs_ddrtree = pd.read_csv("tree_proj_full_branches_plotly_relevant.csv")

color_map = {
    1: 'firebrick', 2: 'gold', 3: 'forestgreen', 4: 'lightseagreen', 5: 'royalblue', 6: 'orchid'
}

branch_map = {
    1: 'Higher risk LBBB (Phenogroup 1)', 2: 'Higher risk LBBB/NSIVCD (Phenogroup 2)', 3: 'Higher risk IVCD (Phenogroup 3)',
    4: 'Average branch RBBB (Phenogroup 4)', 5: 'Lower risk RBBB (Phenogroup 5)', 6: 'Higher risk RBBB (Phenogroup 6)'
}

color_map1 = {
    'firebrick': (178, 34, 34), 'goldenrod': (218, 165, 32), 'forestgreen': (34, 139, 34), 'lightseagreen': (32, 178, 170),
    'royalblue': (65, 105, 225), 'orchid': (218, 112, 214)
}

pheno_tips_map = {
    "pheno1_left1": "Higher risk LBBB (Phenogroup 1): left sub-branch start", 
    "pheno1_left2": "Higher risk LBBB (Phenogroup 1): left sub-branch end", 
    "pheno1_right1": "Higher risk LBBB (Phenogroup 1): right sub-branch start", 
    "pheno1_right2": "Higher risk LBBB (Phenogroup 1): right sub-branch end", 
    "pheno2_1": "Higher risk LBBB/NSIVCD (Phenogroup 2): branch start",
    "pheno2_2": "Higher risk LBBB/NSIVCD (Phenogroup 2): left sub-branch #1",
    "pheno2_31": "Higher risk LBBB/NSIVCD (Phenogroup 2): left sub-branch #2 start", 
    "pheno2_32": "Higher risk LBBB/NSIVCD (Phenogroup 2): left sub-branch #2 end", 
    "pheno2_4": "Higher risk LBBB/NSIVCD (Phenogroup 2): right sub-branch", 
    "pheno2_5": "Higher risk LBBB/NSIVCD (Phenogroup 2): branch end", 
    "pheno3_1": "Higher risk IVCD (Phenogroup 3): branch start", 
    "pheno3_2": "Higher risk IVCD (Phenogroup 3): branch end",
    "pheno3_3": "Higher risk IVCD (Phenogroup 3): right sub-branch",
    "pheno3_41": "Higher risk IVCD (Phenogroup 3): left sub-branch start",
    "pheno3_42": "Higher risk IVCD (Phenogroup 3): left sub-branch end",
    "pheno4_1": "Average branch RBBB (Phenogroup 4) branch end", 
    "pheno4_2": "Average branch RBBB (Phenogroup 4) branch core", 
    "pheno4_3": "Average branch RBBB (Phenogroup 4) branch start", 
    "pheno5_1": "Lower risk RBBB (Phenogroup 5) branch start", 
    "pheno5_2": "Lower risk RBBB (Phenogroup 5) left sub-branch #1",
    "pheno5_31": "Lower risk RBBB (Phenogroup 5) right sub-branch #1 start", 
    "pheno5_32": "Lower risk RBBB (Phenogroup 5) right sub-branch #2 end", 
    "pheno5_4": "Lower risk RBBB (Phenogroup 5) left sub-branch #4", 
    "pheno5_5": "Lower risk RBBB (Phenogroup 5) right sub-branch #2", 
    "pheno5_6": "Lower risk RBBB (Phenogroup 5) branch end", 
    "pheno6_1": "Higher risk RBBB (Phenogroup 6) branch start",
    "pheno6_2": "Higher risk RBBB (Phenogroup 6) branch core", 
    "pheno6_3": "Higher risk RBBB (Phenogroup 6) branch end"
}
 
tips_type_map = {

"Higher risk LBBB (Phenogroup 1): left sub-branch start" : 0, 
"Higher risk LBBB (Phenogroup 1): left sub-branch end" : 1, 
"Higher risk LBBB (Phenogroup 1): right sub-branch start" : 2, 
"Higher risk LBBB (Phenogroup 1): right sub-branch end" : 3, 
"Higher risk LBBB/NSIVCD (Phenogroup 2): branch start" : 4, 
"Higher risk LBBB/NSIVCD (Phenogroup 2): left sub-branch #1" : 5, 
"Higher risk LBBB/NSIVCD (Phenogroup 2): left sub-branch #2 start" : 6, 
"Higher risk LBBB/NSIVCD (Phenogroup 2): left sub-branch #2 end" : 7, 
"Higher risk LBBB/NSIVCD (Phenogroup 2): right sub-branch" : 8, 
"Higher risk LBBB/NSIVCD (Phenogroup 2): branch end" : 9, 
"Higher risk IVCD (Phenogroup 3): branch start" : 10, 
"Higher risk IVCD (Phenogroup 3): branch end" : 11, 
"Higher risk IVCD (Phenogroup 3): right sub-branch" : 12, 
"Higher risk IVCD (Phenogroup 3): left sub-branch start" : 13, 
"Higher risk IVCD (Phenogroup 3): left sub-branch end" : 14, 
"Average branch RBBB (Phenogroup 4) branch end" : 15, 
"Average branch RBBB (Phenogroup 4) branch core" : 16, 
"Average branch RBBB (Phenogroup 4) branch start": 17,  
"Lower risk RBBB (Phenogroup 5) branch start" : 18, 
"Lower risk RBBB (Phenogroup 5) left sub-branch #1": 19, 
"Lower risk RBBB (Phenogroup 5) right sub-branch #1 start" : 20,  
"Lower risk RBBB (Phenogroup 5) right sub-branch #2 end" : 21, 
"Lower risk RBBB (Phenogroup 5) left sub-branch #4" : 22, 
"Lower risk RBBB (Phenogroup 5) right sub-branch #2" : 23, 
"Lower risk RBBB (Phenogroup 5) branch end" : 24, 
"Higher risk RBBB (Phenogroup 6) branch start" : 25, 
"Higher risk RBBB (Phenogroup 6) branch core" : 26, 
"Higher risk RBBB (Phenogroup 6) branch end" : 27

}

color_map_new = {
"Higher risk LBBB (Phenogroup 1): left sub-branch start" : "firebrick", 
"Higher risk LBBB (Phenogroup 1): left sub-branch end" : "firebrick", 
"Higher risk LBBB (Phenogroup 1): right sub-branch start" : "firebrick", 
"Higher risk LBBB (Phenogroup 1): right sub-branch end" : "firebrick", 
"Higher risk LBBB/NSIVCD (Phenogroup 2): branch start" : "goldenrod", 
"Higher risk LBBB/NSIVCD (Phenogroup 2): left sub-branch #1" : "goldenrod", 
"Higher risk LBBB/NSIVCD (Phenogroup 2): left sub-branch #2 start" : "goldenrod", 
"Higher risk LBBB/NSIVCD (Phenogroup 2): left sub-branch #2 end" : "goldenrod", 
"Higher risk LBBB/NSIVCD (Phenogroup 2): right sub-branch" : "goldenrod", 
"Higher risk LBBB/NSIVCD (Phenogroup 2): branch end" : "goldenrod", 
"Higher risk IVCD (Phenogroup 3): branch start" : "forestgreen", 
"Higher risk IVCD (Phenogroup 3): branch end" : "forestgreen", 
"Higher risk IVCD (Phenogroup 3): right sub-branch" : "forestgreen", 
"Higher risk IVCD (Phenogroup 3): left sub-branch start" : "forestgreen", 
"Higher risk IVCD (Phenogroup 3): left sub-branch end" : "forestgreen", 
"Average branch RBBB (Phenogroup 4) branch end" : "lightseagreen", 
"Average branch RBBB (Phenogroup 4) branch core" : "lightseagreen", 
"Average branch RBBB (Phenogroup 4) branch start": "lightseagreen",  
"Lower risk RBBB (Phenogroup 5) branch start" : "royalblue", 
"Lower risk RBBB (Phenogroup 5) left sub-branch #1": "royalblue", 
"Lower risk RBBB (Phenogroup 5) right sub-branch #1 start" : "royalblue",  
"Lower risk RBBB (Phenogroup 5) right sub-branch #2 end" : "royalblue", 
"Lower risk RBBB (Phenogroup 5) left sub-branch #4" : "royalblue", 
"Lower risk RBBB (Phenogroup 5) right sub-branch #2" : "royalblue", 
"Lower risk RBBB (Phenogroup 5) branch end" : "royalblue", 
"Higher risk RBBB (Phenogroup 6) branch start" : "orchid", 
"Higher risk RBBB (Phenogroup 6) branch core" : "orchid", 
"Higher risk RBBB (Phenogroup 6) branch end" : "orchid"

}

broadqrs_ddrtree['phenogroup'] = broadqrs_ddrtree['merged_branchcoords'].map(branch_map)
broadqrs_ddrtree['tips_type_nice'] = broadqrs_ddrtree['tips_type'].map(pheno_tips_map)
broadqrs_ddrtree['tips_type_mapped'] = broadqrs_ddrtree['tips_type_nice'].map(tips_type_map).astype('Int64')

def format_hover_text(text):
    # Insert newline after ':'
    return text.replace(':', ':\n')

broadqrs_ddrtree['formatted_tips_type'] = broadqrs_ddrtree['tips_type_nice'].apply(format_hover_text)


# broadqrs_ddrtree['color'] = broadqrs_ddrtree['merged_branchcoords'].map(color_map).fillna('gray')
broadqrs_ddrtree['color_new'] = broadqrs_ddrtree['tips_type_nice'].map(color_map_new).fillna('gray')

fig = go.Figure()
fig.add_trace(go.Scattergl(
    x=broadqrs_ddrtree['Z1'], y=broadqrs_ddrtree['Z2'], mode='markers', marker=dict(
        size=7, color=broadqrs_ddrtree['color_new'], opacity=0.7, line=dict(width=1, color='black')
    ), name='Scatter Points', hoverinfo='x+y+text', text=broadqrs_ddrtree['tips_type_nice']
))

# fig.add_trace(go.Scattergl(
#     x=broadqrs_ddrtree['Z1'], 
#     y=broadqrs_ddrtree['Z2'], 
#     mode='markers', 
#     marker=dict(
#         size=7, 
#         color=broadqrs_ddrtree['color_new'], 
#         opacity=0.7, 
#         line=dict(width=1, color='black')
#     ), 
#     name='Scatter Points', 
#     hoverinfo='x+y+text', 
#     text=broadqrs_ddrtree['tips_type_nice'],
#     hoverlabel=dict(
#         bgcolor="white",  # Background color of the hover label
#         font_size=12,     # Font size of the text in the hover label
#         align='left',     # Alignment of text within the hover label
#         bordercolor="black",  # Border color of the hover label
#         width=300         # Width of the hover label in pixels
#     )
# ))

fig.update_layout(
    title='Broad QRS DDRTree', xaxis_title='Dimension 1', yaxis_title='Dimension 2',
    width=700, height=700, font=dict(size=15)
)

app.layout = html.Div([
    html.H1(children='Visualising ECGs from the broad QRS DDRTree', style={'textAlign': 'center'}),
    html.Div([
        dcc.Graph(id='scatter-plot', figure=fig),
        dcc.Graph(id='hover-data-plot', style={'margin-top': '1px'})
    ], style={'display': 'flex'})
])

@app.callback(
    Output('hover-data-plot', 'figure'),
    [Input('scatter-plot', 'hoverData')]
)
def update_hover_plot(hoverData):
    if hoverData is None:
        return {
            'data': [], 'layout': {
                'annotations': [{'x': 3, 'y': 1.5, 'text': "Hover over a point on the tree to see the reconstructed ECG.", 'showarrow': False, 'font': {'size': 13, 'color': 'black'}}],
                'xaxis': {'visible': False}, 'yaxis': {'visible': False}, 'width': 400, 'height': 600
            }
        }

    point_index = hoverData['points'][0]['pointIndex']
    tips_type = broadqrs_ddrtree.iloc[point_index]['tips_type_mapped']
    ecg_data = decoded_ecgs_array_12L[tips_type]

    lead_names = ['I', 'aVR', 'V1', 'V4', 'II', 'aVL', 'V2', 'V5', 'III', 'aVF', 'V3', 'V6']
    fig_plotly = plot_ecg_plotly(ecg_data, 400, lead_names=lead_names, subplot_shape=(3, 4), ylim=(-2, 2), subplots=True, figsize=(1000, 1000), std=decoded_stds_array_12L[tips_type])

    phenogroup = broadqrs_ddrtree['formatted_tips_type'].iloc[point_index]
    point_color_name = hoverData['points'][0]['marker.color']
    rgb_color = color_map1.get(point_color_name)
    plotly_color = f'rgb{rgb_color}'
    fig_plotly['layout']['title'].update(text=f"Reconstructed ECG - {phenogroup}", font=dict(color=plotly_color, size=15))

    return fig_plotly

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=8050)
