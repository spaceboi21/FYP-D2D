import dash
import base64
import io
from dash import dcc, html, Input, Output
import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class

# Initialize the Dash app
app = dash.Dash(__name__)

# Layout of the app
app.layout = html.Div([
    html.H1("Automated CSV Data Visualization with AutoViz"),
    
    # File upload component
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select a CSV File')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    
    # Dropdown for selecting visualization type
    dcc.Dropdown(
        id='visualization-type',
        options=[
            {'label': 'Bar Chart', 'value': 'bar'},
            {'label': 'Pie Chart', 'value': 'pie'},
            {'label': 'Scatter Plot', 'value': 'scatter'},
            {'label': 'Line Plot', 'value': 'line'}
        ],
        placeholder="Select a visualization type",
        style={'width': '50%', 'margin': '10px'}
    ),
    
    # Div to hold the AutoViz graphs
    html.Div(id='autoviz-graphs', style={'width': '100%'})
])

# Function to parse and clean uploaded data
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    
    # Data cleaning steps
    df.dropna(inplace=True)  # Handle NULL values
    df.drop_duplicates(inplace=True)  # Drop duplicates
    
    return df

# Callback to handle file upload and initiate AutoViz
@app.callback(
    Output('autoviz-graphs', 'children'),
    [Input('upload-data', 'contents'),
     Input('visualization-type', 'value')],
    prevent_initial_call=True
)
def update_output(contents, viz_type):
    if contents is not None:
        df = parse_contents(contents)
        AV = AutoViz_Class()
        # Using AutoViz to automatically generate visualizations
        figure = AV.AutoViz(
            "",
            depVar='',
            dfte=df,
            header=0,
            verbose=2,
            lowess=False,
            chart_format='svg',  # Using 'svg' for web display compatibility
            max_rows_analyzed=150000,
            max_cols_analyzed=30,
        )
        return figure
    return "Please upload a file to see visualizations."

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
