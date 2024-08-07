# import dash
# import base64
# import io
# import pandas as pd
# import plotly.express as px
# import openai
# from openai import OpenAI
# import os
# import logging
# import dash_bootstrap_components as dbc
# from dash import dcc, html, Input, Output, State
# from flask_caching import Cache

# os.environ['OPENAI_API_KEY'] = 'sk-proj-2TXaScJW88nTcTe3PoaHT3BlbkFJ0nwAuBbo7VbxdxZmHq8b'
# client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# # Initialize the Dash app with callback exception suppression
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
# cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache'})

# app.layout = html.Div([
#     html.H1("AI-Enhanced Multi-Graph CSV Data Visualization Dashboard", style={'text-align': 'center'}),
#     dcc.Upload(
#         id='upload-data',
#         children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
#         style={
#             'width': '100%', 'height': '60px', 'lineHeight': '60px',
#             'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
#             'textAlign': 'center', 'margin': '10px'
#         },
#         multiple=False
#     ),
#     html.Div(id='color-picker-container', style={'display': 'none'}, children=[
#         html.Label('Select Graph Color:'),
#         dcc.Input(
#             id='color-picker', 
#             type='text', 
#             value='#636EFA', 
#             style={'margin-bottom': '10px'}
#         )
#     ]),
#     html.Button('Update Color', id='update-color', n_clicks=0, style={'display': 'none'}),
#     dbc.Container(id='graphs-container', fluid=True)
# ])

# def parse_contents(contents):
#     logging.info("Parsing contents")
#     if contents:
#         content_type, content_string = contents.split(',')
#         decoded = base64.b64decode(content_string)
#         encodings = ['utf-8', 'latin1', 'ISO-8859-1']
#         for encoding in encodings:
#             try:
#                 df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
#                 df = handle_missing_values(df)
#                 return df
#             except UnicodeDecodeError:
#                 continue
#         return None
#     return None

# def handle_missing_values(df):
#     logging.info("Handling missing values")
#     for column in df.columns:
#         if pd.api.types.is_numeric_dtype(df[column]):
#             df[column] = df[column].fillna(0)
#         else:
#             df[column] = df[column].fillna('Nil')
#     return df

# def generate_prompt(df):
#     column_details = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
#     prompt = f"""Given a dataset with columns {column_details}, I need recommendations for visualizations that effectively showcase the data. First, thoroughly understand the schema by reviewing the column names. Based on this understanding, determine which columns pair well for compelling visualizations. Please follow these specific guidelines for your response:
#     - Provide visualization types one at a time.
#     - For each visualization, specify the type and the best columns to use.
#     - Your response should be formatted as follows for each visualization suggestion:
#     - First line: viz_type = [visualization type in lowercase]
#     - Second line: columns = [list of column names]
#     - Do not include any additional text, notes, or explanations in your response.
#     - Use only these types of graphs: heatmap, pie, histogram, area, box, line, scatter, multi_line, funnel, sunburst, treemap, icicle, violin, ecdf, strip, iris, scatter_geo, scatter_polar, line_polar, bar_polar, scatter_3d.
#     Here is an example format for clarity:
#     viz_type = area
#     columns = ['Number of Records', 'I feel safe on campus']
#     Ensure that your response maintains this format consistently throughout as our parsing function depends on it. Suggest visualizations that make the most sense based on the data schema and column relationships."""

    
#     return prompt

# @cache.memoize(timeout=300)
# def query_openai(prompt):
#     """ Send prompt to OpenAI and return the suggested visualization """
    
    
#     client = OpenAI()

#     response = client.chat.completions.create(
#         model="gpt-4",
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         max_tokens=200
#     )
    
#     print(response.choices[0].message.content.strip())  # Debugging: See what OpenAI suggests
#     return parse_ai_response(response.choices[0].message.content.strip())

# @app.callback(
#     [Output('color-picker-container', 'style'),
#      Output('update-color', 'style'),
#      Output('graphs-container', 'children')],
#     Input('upload-data', 'contents')
# )
# def update_graph(contents):
#     logging.info("Updating graph")
#     if contents:
#         df = parse_contents(contents)
#         if df is not None:
#             prompt = generate_prompt(df)
#             visualization_suggestions = query_openai(prompt)
#             figures = []
#             rows = []

#             for i, viz in enumerate(visualization_suggestions):
#                 logging.info(f"Creating figure {i + 1}")
#                 fig = create_figure(df, viz)
#                 if fig is not None:
#                     figures.append(fig)

#             for i in range(0, len(figures), 2):
#                 row = dbc.Row([
#                     dbc.Col(dcc.Graph(figure=figures[i], id=f'graph-output{i+1}'), width=6) if i < len(figures) else None,
#                     dbc.Col(dcc.Graph(figure=figures[i + 1], id=f'graph-output{i+2}'), width=6) if i + 1 < len(figures) else None
#                 ])
#                 rows.append(row)
            
#             return {'display': 'block'}, {'display': 'block'}, rows
    
#     return {'display': 'none'}, {'display': 'none'}, []

# @app.callback(
#     [Output(f'graph-output{i}', 'figure') for i in range(1, 9)],
#     [Input('upload-data', 'contents'),
#      Input('update-color', 'n_clicks')],
#     State('color-picker', 'value')
# )
# def update_graph_with_color(contents, n_clicks, color):
#     logging.info("Updating graph with color")
#     figures = [{} for _ in range(8)]  # Initialize an array of empty figures
#     if contents:
#         df = parse_contents(contents)
#         if df is not None:
#             prompt = generate_prompt(df)
#             visualization_suggestions = query_openai(prompt)
#             # Generate figures for each visualization suggestion
#             for i, viz in enumerate(visualization_suggestions):
#                 if i < 8:  # Only populate up to eight figures
#                     logging.info(f"Creating colored figure {i + 1}")
#                     figures[i] = create_figure(df, viz, color=color)
#     return figures

# def parse_ai_response(response):
#     visualizations = []
#     current_viz_type = None
#     lines = response.strip().split('\n')
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
#         if "viz_type" in line:
#             try:
#                 current_viz_type = line.split('=')[1].strip().strip("'")
#             except IndexError:
#                 logging.warning(f"Malformed viz_type line: {line}")
#                 current_viz_type = None
#         elif "columns" in line:
#             try:
#                 columns_part = line.split('=')[1].strip()
#                 columns = [col.strip().strip('[]').replace("'", '') for col in columns_part.split(",")]
#                 if current_viz_type and columns:
#                     visualizations.append({'type': current_viz_type, 'columns': columns})
#                     current_viz_type = None
#             except IndexError:
#                 logging.warning(f"Malformed columns line: {line}")
#         else:
#             logging.warning(f"Skipping unrecognized line: {line}")
#     return visualizations

# logging.basicConfig(level=logging.INFO)

# # def create_figure(df, viz_info, color=None):
# #     fig = None
# #     try:
# #         if viz_info and 'type' in viz_info and 'columns' in viz_info:
# #             columns_exist = all(col in df.columns for col in viz_info['columns'])
# #             if not columns_exist:
# #                 logging.warning(f"One or more columns in {viz_info['columns']} do not exist in the dataframe.")
# #                 return None

# #             graph_kwargs = {'color_discrete_sequence': [color] if color else None}
# #             if viz_info['type'] == 'scatter' and len(viz_info['columns']) >= 2:
# #                 fig = px.scatter(df, x=viz_info['columns'][0], y=viz_info['columns'][1], **graph_kwargs)
# #             elif viz_info['type'] == 'line' and len(viz_info['columns']) >= 2:
# #                 fig = px.line(df, x=viz_info['columns'][0], y=viz_info['columns'][1], **graph_kwargs)
# #             elif viz_info['type'] == 'area' and len(viz_info['columns']) >= 2:
# #                 fig = px.area(df, x=viz_info['columns'][0], y=viz_info['columns'][1], **graph_kwargs)
# #             elif viz_info['type'] == 'bar' and len(viz_info['columns']) >= 2:
# #                 fig = px.bar(df, x=viz_info['columns'][0], y=viz_info['columns'][1], **graph_kwargs)
# #             elif viz_info['type'] == 'histogram':
# #                 fig = px.histogram(df, x=viz_info['columns'][0], **graph_kwargs)
# #             elif viz_info['type'] == 'pie':
# #                 if len(viz_info['columns']) > 1:
# #                     fig = px.pie(df, names=viz_info['columns'][0], values=viz_info['columns'][1], **graph_kwargs)
# #                 else:
# #                     fig = px.pie(df, names=viz_info['columns'][0], **graph_kwargs)
# #             elif viz_info['type'] == 'box':
# #                 fig = px.box(df, y=viz_info['columns'][0], **graph_kwargs)
# #             if fig and color:
# #                 fig.update_layout(
# #                     paper_bgcolor='rgba(0,0,0,0)',
# #                     plot_bgcolor='rgba(0,0,0,0)',
# #                     font_color=color,
# #                 )
# #     except Exception as e:
# #         logging.error(f"Error in creating figure: {e}")
# #     return fig

# import seaborn as sns

# def create_figure(df, viz_info, color=None):
#     fig = None
#     try:
#         if viz_info and 'type' in viz_info and 'columns' in viz_info:
#             columns_exist = all(col in df.columns for col in viz_info['columns'])
#             if not columns_exist:
#                 logging.warning(f"One or more columns in {viz_info['columns']} do not exist in the dataframe.")
#                 return None

#             # Define Seaborn color palettes
#             color_palettes = {
#                 'scatter': sns.color_palette("husl", 10).as_hex(),
#                 'line': sns.color_palette("viridis", 10).as_hex(),
#                 'area': sns.color_palette("cubehelix", 10).as_hex(),
#                 'bar': sns.color_palette("muted", 10).as_hex(),
#                 'histogram': sns.color_palette("pastel", 10).as_hex(),
#                 'pie': sns.color_palette("bright", 10).as_hex(),
#                 'box': sns.color_palette("dark", 10).as_hex(),
#                 'heatmap': sns.color_palette("coolwarm", as_cmap=True),
#                 'multi_line': sns.color_palette("viridis", 10).as_hex(),
#                 'funnel': sns.color_palette("icefire", 10).as_hex(),
#                 'sunburst': sns.color_palette("rocket", 10).as_hex(),
#                 'treemap': sns.color_palette("mako", 10).as_hex(),
#                 'icicle': sns.color_palette("flare", 10).as_hex(),
#                 'violin': sns.color_palette("ch:s=.25,rot=-.25", 10).as_hex(),
#                 'ecdf': sns.color_palette("crest", 10).as_hex(),
#                 'strip': sns.color_palette("twilight", 10).as_hex(),
#                 'iris': sns.color_palette("Set2", 10).as_hex(),
#                 'scatter_geo': sns.color_palette("mako", 10).as_hex(),
#                 'scatter_polar': sns.color_palette("Paired", 10).as_hex(),
#                 'line_polar': sns.color_palette("PuBuGn", 10).as_hex(),
#                 'bar_polar': sns.color_palette("YlGnBu", 10).as_hex(),
#                 'scatter_3d': sns.color_palette("magma", 10).as_hex()
#             }

#             palette = color_palettes.get(viz_info['type'], sns.color_palette("husl", 10).as_hex())

#             if viz_info['type'] == 'scatter' and len(viz_info['columns']) >= 2:
#                 fig = px.scatter(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'line' and len(viz_info['columns']) >= 2:
#                 fig = px.line(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None, color_discrete_sequence=palette)
#             elif viz_info['type'] == 'area' and len(viz_info['columns']) >= 2:
#                 fig = px.area(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'bar' and len(viz_info['columns']) >= 2:
#                 fig = px.bar(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'histogram':
#                 fig = px.histogram(df, x=viz_info['columns'][0], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'pie':
#                 fig = px.pie(df, names=viz_info['columns'][0], values=viz_info['columns'][1] if len(viz_info['columns']) > 1 else None, color_discrete_sequence=palette)
#             elif viz_info['type'] == 'box':
#                 fig = px.box(df, y=viz_info['columns'][0], color=viz_info['columns'][1] if len(viz_info['columns']) > 1 else None, color_discrete_sequence=palette)
#             elif viz_info['type'] == 'heatmap' and len(viz_info['columns']) >= 2:
#                 fig = px.density_heatmap(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_continuous_scale=palette)
#             elif viz_info['type'] == 'multi_line' and len(viz_info['columns']) >= 3:
#                 fig = px.line(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color=viz_info['columns'][2], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'funnel' and len(viz_info['columns']) >= 2:
#                 fig = px.funnel(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'sunburst' and len(viz_info['columns']) >= 3:
#                 fig = px.sunburst(df, path=viz_info['columns'], values=viz_info['columns'][2], color_continuous_scale=palette)
#             elif viz_info['type'] == 'treemap' and len(viz_info['columns']) >= 3:
#                 fig = px.treemap(df, path=viz_info['columns'], values=viz_info['columns'][2], color_continuous_scale=palette)
#             elif viz_info['type'] == 'icicle' and len(viz_info['columns']) >= 3:
#                 fig = px.icicle(df, path=viz_info['columns'], values=viz_info['columns'][2], color_continuous_scale=palette)
#             elif viz_info['type'] == 'violin' and len(viz_info['columns']) >= 2:
#                 fig = px.violin(df, y=viz_info['columns'][0], x=viz_info['columns'][1], color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None, box=True, points="all", color_discrete_sequence=palette)
#             elif viz_info['type'] == 'ecdf' and len(viz_info['columns']) >= 1:
#                 fig = px.ecdf(df, x=viz_info['columns'][0], color=viz_info['columns'][1] if len(viz_info['columns']) > 1 else None, color_discrete_sequence=palette)
#             elif viz_info['type'] == 'strip' and len(viz_info['columns']) >= 2:
#                 fig = px.strip(df, y=viz_info['columns'][0], x=viz_info['columns'][1], color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None, color_discrete_sequence=palette)
#             elif viz_info['type'] == 'iris' and len(viz_info['columns']) >= 2:
#                 fig = px.scatter(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None, color_discrete_sequence=palette)
#             elif viz_info['type'] == 'scatter_geo' and len(viz_info['columns']) >= 2:
#                 fig = px.scatter_geo(df, lat=viz_info['columns'][0], lon=viz_info['columns'][1], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'scatter_polar' and len(viz_info['columns']) >= 2:
#                 fig = px.scatter_polar(df, r=viz_info['columns'][0], theta=viz_info['columns'][1], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'line_polar' and len(viz_info['columns']) >= 2:
#                 fig = px.line_polar(df, r=viz_info['columns'][0], theta=viz_info['columns'][1], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'bar_polar' and len(viz_info['columns']) >= 2:
#                 fig = px.bar_polar(df, r=viz_info['columns'][0], theta=viz_info['columns'][1], color_discrete_sequence=palette)
#             elif viz_info['type'] == 'scatter_3d' and len(viz_info['columns']) >= 3:
#                 fig = px.scatter_3d(df, x=viz_info['columns'][0], y=viz_info['columns'][1], z=viz_info['columns'][2], color_discrete_sequence=palette)

#             if fig:
#                 fig.update_layout(
#                     paper_bgcolor='rgba(0,0,0,0)',
#                     plot_bgcolor='rgba(0,0,0,0)',
#                 )
#     except Exception as e:
#         logging.error(f"Error in creating figure: {e}")
#     return fig





# if __name__ == '__main__':
#     app.run_server(debug=True)


import dash
import base64
import io
import pandas as pd
import plotly.express as px
import openai
from openai import OpenAI
import os
import logging
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from flask_caching import Cache

os.environ['OPENAI_API_KEY'] = 'sk-proj-2TXaScJW88nTcTe3PoaHT3BlbkFJ0nwAuBbo7VbxdxZmHq8b'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Initialize the Dash app with callback exception suppression
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache'})

app.layout = html.Div([
    html.H1("AI-Enhanced Multi-Graph CSV Data Visualization Dashboard", style={'text-align': 'center'}),
    dcc.Upload(
        id='upload-data',
        children=html.Div(['Drag and Drop or ', html.A('Select a CSV File')]),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
            'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='color-picker-container', style={'display': 'none'}, children=[
        html.Label('Select Graph Color:'),
        dcc.Input(
            id='color-picker', 
            type='text', 
            value='#636EFA', 
            style={'margin-bottom': '10px'}
        )
    ]),
    html.Button('Update Color', id='update-color', n_clicks=0, style={'display': 'none'}),
    dbc.Container(id='graphs-container', fluid=True)
])

def parse_contents(contents):
    logging.info("Parsing contents")
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        encodings = ['utf-8', 'latin1', 'ISO-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
                df = handle_missing_values(df)
                return df
            except UnicodeDecodeError:
                continue
        return None
    return None

def handle_missing_values(df):
    logging.info("Handling missing values")
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(0)
        else:
            df[column] = df[column].fillna('Nil')
    return df

def generate_prompt(df):
    column_details = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
    prompt = f"""Given a dataset with columns {column_details}, I need recommendations for visualizations that effectively showcase the data. First, thoroughly understand the schema by reviewing the column names. Based on this understanding, determine which columns pair well for compelling visualizations. Please follow these specific guidelines for your response:
    - Provide visualization types one at a time.
    - For each visualization, specify the type and the best columns to use.
    - Your response should be formatted as follows for each visualization suggestion:
    - First line: viz_type = [visualization type in lowercase]
    - Second line: columns = [list of column names]
    - Do not include any additional text, notes, or explanations in your response.
    - Use only these types of graphs: heatmap, pie, histogram, area, box, line, scatter, multi_line, funnel, sunburst, treemap, icicle, violin, ecdf, strip, iris, scatter_geo, scatter_polar, line_polar, bar_polar, scatter_3d.
    Here is an example format for clarity:
    viz_type = area
    columns = ['Number of Records', 'I feel safe on campus']
    Ensure that your response maintains this format consistently throughout as our parsing function depends on it. Suggest visualizations that make the most sense based on the data schema and column relationships."""
    
    return prompt

@cache.memoize(timeout=300)
def query_openai(prompt):
    """ Send prompt to OpenAI and return the suggested visualization """
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    
    print(response.choices[0].message.content.strip())  # Debugging: See what OpenAI suggests
    return parse_ai_response(response.choices[0].message.content.strip())

@app.callback(
    [Output('color-picker-container', 'style'),
     Output('update-color', 'style'),
     Output('graphs-container', 'children')],
    Input('upload-data', 'contents')
)
def update_graph(contents):
    logging.info("Updating graph")
    if contents:
        df = parse_contents(contents)
        if df is not None:
            prompt = generate_prompt(df)
            visualization_suggestions = query_openai(prompt)
            figures = []
            rows = []

            for i, viz in enumerate(visualization_suggestions):
                logging.info(f"Creating figure {i + 1}")
                fig = create_figure(df, viz)
                if fig is not None:
                    figures.append(fig)

            for i in range(0, len(figures), 2):
                row = dbc.Row([
                    dbc.Col(dcc.Graph(figure=figures[i], id=f'graph-output{i+1}'), width=6) if i < len(figures) else None,
                    dbc.Col(dcc.Graph(figure=figures[i + 1], id=f'graph-output{i+2}'), width=6) if i + 1 < len(figures) else None
                ])
                rows.append(row)
            
            return {'display': 'block'}, {'display': 'block'}, rows
    
    return {'display': 'none'}, {'display': 'none'}, []

@app.callback(
    [Output(f'graph-output{i}', 'figure') for i in range(1, 9)],
    [Input('upload-data', 'contents'),
     Input('update-color', 'n_clicks')],
    State('color-picker', 'value')
)
def update_graph_with_color(contents, n_clicks, color):
    logging.info("Updating graph with color")
    figures = [{} for _ in range(8)]  # Initialize an array of empty figures
    if contents:
        df = parse_contents(contents)
        if df is not None:
            prompt = generate_prompt(df)
            visualization_suggestions = query_openai(prompt)
            # Generate figures for each visualization suggestion
            for i, viz in enumerate(visualization_suggestions):
                if i < 8:  # Only populate up to eight figures
                    logging.info(f"Creating colored figure {i + 1}")
                    figures[i] = create_figure(df, viz, color=color)
    return figures

def parse_ai_response(response):
    visualizations = []
    current_viz_type = None
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "viz_type" in line:
            try:
                current_viz_type = line.split('=')[1].strip().strip("'")
            except IndexError:
                logging.warning(f"Malformed viz_type line: {line}")
                current_viz_type = None
        elif "columns" in line:
            try:
                columns_part = line.split('=')[1].strip()
                columns = [col.strip().strip('[]').replace("'", '') for col in columns_part.split(",")]
                if current_viz_type and columns:
                    visualizations.append({'type': current_viz_type, 'columns': columns})
                    current_viz_type = None
            except IndexError:
                logging.warning(f"Malformed columns line: {line}")
        else:
            logging.warning(f"Skipping unrecognized line: {line}")
    return visualizations

logging.basicConfig(level=logging.INFO)

import seaborn as sns

def create_figure(df, viz_info, color=None):
    fig = None
    try:
        if viz_info and 'type' in viz_info and 'columns' in viz_info:
            columns_exist = all(col in df.columns for col in viz_info['columns'])
            if not columns_exist:
                logging.warning(f"One or more columns in {viz_info['columns']} do not exist in the dataframe.")
                return None

            # Define Seaborn color palettes
            color_palettes = {
                'scatter': sns.color_palette("husl", 10).as_hex(),
                'line': sns.color_palette("viridis", 10).as_hex(),
                'area': sns.color_palette("cubehelix", 10).as_hex(),
                'bar': sns.color_palette("muted", 10).as_hex(),
                'histogram': sns.color_palette("pastel", 10).as_hex(),
                'pie': sns.color_palette("bright", 10).as_hex(),
                'box': sns.color_palette("dark", 10).as_hex(),
                'heatmap': sns.color_palette("coolwarm", as_cmap=True),
                'multi_line': sns.color_palette("viridis", 10).as_hex(),
                'funnel': sns.color_palette("icefire", 10).as_hex(),
                'sunburst': sns.color_palette("rocket", 10).as_hex(),
                'treemap': sns.color_palette("mako", 10).as_hex(),
                'icicle': sns.color_palette("flare", 10).as_hex(),
                'violin': sns.color_palette("ch:s=.25,rot=-.25", 10).as_hex(),
                'ecdf': sns.color_palette("crest", 10).as_hex(),
                'strip': sns.color_palette("twilight", 10).as_hex(),
                'iris': sns.color_palette("Set2", 10).as_hex(),
                'scatter_geo': sns.color_palette("mako", 10).as_hex(),
                'scatter_polar': sns.color_palette("Paired", 10).as_hex(),
                'line_polar': sns.color_palette("PuBuGn", 10).as_hex(),
                'bar_polar': sns.color_palette("YlGnBu", 10).as_hex(),
                'scatter_3d': sns.color_palette("magma", 10).as_hex()
            }

            palette = color_palettes.get(viz_info['type'], sns.color_palette("husl", 10).as_hex())

            title = f"{viz_info['type'].capitalize()} of {' and '.join(viz_info['columns'])}"

            if viz_info['type'] == 'scatter' and len(viz_info['columns']) >= 2:
                fig = px.scatter(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'line' and len(viz_info['columns']) >= 2:
                fig = px.line(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None, color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'area' and len(viz_info['columns']) >= 2:
                fig = px.area(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'bar' and len(viz_info['columns']) >= 2:
                fig = px.bar(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'histogram':
                fig = px.histogram(df, x=viz_info['columns'][0], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'pie':
                fig = px.pie(df, names=viz_info['columns'][0], values=viz_info['columns'][1] if len(viz_info['columns']) > 1 else None, color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'box':
                fig = px.box(df, y=viz_info['columns'][0], color=viz_info['columns'][1] if len(viz_info['columns']) > 1 else None, color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'heatmap' and len(viz_info['columns']) >= 2:
                fig = px.density_heatmap(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_continuous_scale=palette, title=title)
            elif viz_info['type'] == 'multi_line' and len(viz_info['columns']) >= 3:
                fig = px.line(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color=viz_info['columns'][2], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'funnel' and len(viz_info['columns']) >= 2:
                fig = px.funnel(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'sunburst' and len(viz_info['columns']) >= 3:
                fig = px.sunburst(df, path=viz_info['columns'], values=viz_info['columns'][2], color_continuous_scale=palette, title=title)
            elif viz_info['type'] == 'treemap' and len(viz_info['columns']) >= 3:
                fig = px.treemap(df, path=viz_info['columns'], values=viz_info['columns'][2], color_continuous_scale=palette, title=title)
            elif viz_info['type'] == 'icicle' and len(viz_info['columns']) >= 3:
                fig = px.icicle(df, path=viz_info['columns'], values=viz_info['columns'][2], color_continuous_scale=palette, title=title)
            elif viz_info['type'] == 'violin' and len(viz_info['columns']) >= 2:
                fig = px.violin(df, y=viz_info['columns'][0], x=viz_info['columns'][1], color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None, box=True, points="all", color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'ecdf' and len(viz_info['columns']) >= 1:
                fig = px.ecdf(df, x=viz_info['columns'][0], color=viz_info['columns'][1] if len(viz_info['columns']) > 1 else None, color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'strip' and len(viz_info['columns']) >= 2:
                fig = px.strip(df, y=viz_info['columns'][0], x=viz_info['columns'][1], color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None, color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'iris' and len(viz_info['columns']) >= 2:
                fig = px.scatter(df, x=viz_info['columns'][0], y=viz_info['columns'][1], color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None, color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'scatter_geo' and len(viz_info['columns']) >= 2:
                fig = px.scatter_geo(df, lat=viz_info['columns'][0], lon=viz_info['columns'][1], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'scatter_polar' and len(viz_info['columns']) >= 2:
                fig = px.scatter_polar(df, r=viz_info['columns'][0], theta=viz_info['columns'][1], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'line_polar' and len(viz_info['columns']) >= 2:
                fig = px.line_polar(df, r=viz_info['columns'][0], theta=viz_info['columns'][1], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'bar_polar' and len(viz_info['columns']) >= 2:
                fig = px.bar_polar(df, r=viz_info['columns'][0], theta=viz_info['columns'][1], color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'scatter_3d' and len(viz_info['columns']) >= 3:
                fig = px.scatter_3d(df, x=viz_info['columns'][0], y=viz_info['columns'][1], z=viz_info['columns'][2], color_discrete_sequence=palette, title=title)

            if fig:
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                )
    except Exception as e:
        logging.error(f"Error in creating figure: {e}")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
