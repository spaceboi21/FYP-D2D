import dash
import base64
import io
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.express as px
from openai import OpenAI
import os
import logging
import dash_daq as daq

os.environ['OPENAI_API_KEY'] = 'sk-proj-82Wqg7vp4qdFebUFtDnlT3BlbkFJHEkBCJDfbVkylPfkmuyK'
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Initialize the Dash app
app = dash.Dash(__name__)

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
    html.Div([
        html.Div([dcc.Graph(id='graph-output1'), daq.ColorPicker(id='color-picker1', value={"hex": "#119DFF"})], className="three columns"),
        html.Div([dcc.Graph(id='graph-output2'), daq.ColorPicker(id='color-picker2', value={"hex": "#119DFF"})], className="three columns"),
        html.Div([dcc.Graph(id='graph-output3'), daq.ColorPicker(id='color-picker3', value={"hex": "#119DFF"})], className="three columns"),
        html.Div([dcc.Graph(id='graph-output4'), daq.ColorPicker(id='color-picker4', value={"hex": "#119DFF"})], className="three columns"),
    ], className="row"),
    html.Div([
        html.Div([dcc.Graph(id='graph-output5'), daq.ColorPicker(id='color-picker5', value={"hex": "#119DFF"})], className="three columns"),
        html.Div([dcc.Graph(id='graph-output6'), daq.ColorPicker(id='color-picker6', value={"hex": "#119DFF"})], className="three columns"),
        html.Div([dcc.Graph(id='graph-output7'), daq.ColorPicker(id='color-picker7', value={"hex": "#119DFF"})], className="three columns"),
        html.Div([dcc.Graph(id='graph-output8'), daq.ColorPicker(id='color-picker8', value={"hex": "#119DFF"})], className="three columns"),
    ], className="row")
])


def parse_contents(contents):
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        return df
    return None

def generate_prompt(df):
    """ Generate a detailed prompt for OpenAI based on dataframe's schema """
    column_details = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
    prompt = f"Given a dataset with columns {column_details}, what are the best visualizations to showcase the data effectively? Suggest types of graphs and the best columns to use for each. Go step by step, do not decide on the visualisation unless you have understood the schema. You should understand the schema properly and study the column names and then decide what columns would go best together in creating a good visualisation, Give your answers in this format,      viz_type = lines[0].strip().lower() columns = [line.strip() for line in lines[1:]] return type: viz_type, columns: columns this is my parsing code, this is how I should be able to get out the responses so that they can be used. Make sure you response consists only the visualisations and no other text at all no notes nothing only the visualisations and the columns in the format asked. Give response in one format only everytime you are prompted to. You can use the graphs heatmap,pie,histogram,area,box,line,scatter only. And give the names of both columns like how you did before."
    return prompt

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
    [Output('graph-output1', 'figure'), Output('graph-output2', 'figure'),
     Output('graph-output3', 'figure'), Output('graph-output4', 'figure'),
     Output('graph-output5', 'figure'), Output('graph-output6', 'figure'),
     Output('graph-output7', 'figure'), Output('graph-output8', 'figure')],
    [Input('upload-data', 'contents'),
     Input('color-picker1', 'value'), Input('color-picker2', 'value'),
     Input('color-picker3', 'value'), Input('color-picker4', 'value'),
     Input('color-picker5', 'value'), Input('color-picker6', 'value'),
     Input('color-picker7', 'value'), Input('color-picker8', 'value')],
    prevent_initial_call=True
)
def update_graph(contents, color1, color2, color3, color4, color5, color6, color7, color8):
    colors = [color1, color2, color3, color4, color5, color6, color7, color8]
    figures = [{}] * 8  # Initialize an array of empty figures
    if contents:
        df = parse_contents(contents)
        if df is not None:
            prompt = generate_prompt(df)
            visualization_suggestions = query_openai(prompt)
            # Generate figures for each visualization suggestion
            for i, viz in enumerate(visualization_suggestions):
                if i < 8:  # Only populate up to eight figures
                    figures[i] = create_figure(df, viz, color=colors[i]['hex'])
    return figures



def parse_ai_response(response):
    """
    Parse OpenAI's response to handle visualization type and columns described on separate lines.
    
    Args:
    response (str): A string response from OpenAI API, expected to contain alternating lines of
                    visualization types and columns, with possibly blank lines in between.

    Returns:
    list of dicts: Each dict contains 'type' with the visualization type and 'columns' with a list of columns.
    """
    # Initialize variables to hold the current visualization type and the list of visualizations
    visualizations = []
    current_viz_type = None
    
    # Split the response into lines
    lines = response.strip().split('\n')
    
    # Iterate over each line to parse it
    for line in lines:
        line = line.strip()
        if not line:
            continue  # Skip empty lines
        
        # Determine if the line is specifying a visualization type or columns
        if "viz_type" in line:
            # Extract the visualization type from the line
            try:
                current_viz_type = line.split('=')[1].strip().strip("'")
            except IndexError:
                logging.warning(f"Malformed viz_type line: {line}")
                current_viz_type = None
        elif "columns" in line:
            # Extract the columns from the line
            try:
                columns_part = line.split('=')[1].strip()
                columns = [col.strip().strip('[]').replace("'", '') for col in columns_part.split(",")]
                if current_viz_type and columns:
                    visualizations.append({'type': current_viz_type, 'columns': columns})
                    current_viz_type = None  # Reset current visualization type after using it
            except IndexError:
                logging.warning(f"Malformed columns line: {line}")
        else:
            logging.warning(f"Skipping unrecognized line: {line}")
    
    # Return the list of parsed visualizations
    return visualizations

# Enable basic logging at the INFO level
logging.basicConfig(level=logging.INFO)



import plotly.express as px

def create_figure(df, viz_info, color=None):
    """
    Create a Plotly figure based on visualization info and color choice.

    Args:
        df (pd.DataFrame): The dataframe containing the data.
        viz_info (dict): Dictionary with keys 'type' and 'columns', where 'type' is the type of
                         plot and 'columns' is a list of column names to be used in the plot.
        color (str): Hex code for the color to be used in the plot.

    Returns:
        plotly.graph_objs._figure.Figure: A Plotly figure object configured according to viz_info.
    """
    fig = None
    if viz_info and 'type' in viz_info and 'columns' in viz_info:
        # General argument for all graph types that support color customization
        graph_kwargs = {'color_discrete_sequence': [color] if color else None}

        # Define graph based on type
        if viz_info['type'] == 'scatter':
            fig = px.scatter(df, x=viz_info['columns'][0], y=viz_info['columns'][1], **graph_kwargs)
        
        elif viz_info['type'] == 'bar':
            fig = px.bar(df, x=viz_info['columns'][0], y=viz_info['columns'][1], **graph_kwargs)
        
        elif viz_info['type'] == 'line':
            fig = px.line(df, x=viz_info['columns'][0], y=viz_info['columns'][1], **graph_kwargs)
        
        elif viz_info['type'] == 'histogram':
            fig = px.histogram(df, x=viz_info['columns'][0], **graph_kwargs)
        
        elif viz_info['type'] == 'pie':
            # Pie charts typically use the names for labels and another column for values
            if len(viz_info['columns']) > 1:
                fig = px.pie(df, names=viz_info['columns'][0], values=viz_info['columns'][1], **graph_kwargs)
            else:
                fig = px.pie(df, names=viz_info['columns'][0], **graph_kwargs)
        
        elif viz_info['type'] == 'box':
            fig = px.box(df, y=viz_info['columns'][0], **graph_kwargs)
        
        elif viz_info['type'] == 'area':
            fig = px.area(df, x=viz_info['columns'][0], y=viz_info['columns'][1], **graph_kwargs)

        # Add more visualization types here as needed

        # Additional settings for the figure
        if fig and color:
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
                plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot background
                font_color=color,  # Set font color
            )
    
    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
