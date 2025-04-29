"""
dash_flask_app.py

Flask+Dash app:
- Auth (login/signup)
- CSV Upload + local analysis
- Button to Ingest CSV to Pinecone (calls /upload_csv)
- Chat input to ask GPT-based questions about the ingested data (calls /ask)
"""

import os
import logging
import base64
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dotenv

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State

from flask import Flask, render_template, redirect, url_for, request, session, send_file
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, UserMixin, login_user, logout_user, login_required, current_user
)
from flask_caching import Cache
from werkzeug.security import generate_password_hash, check_password_hash
from openai import OpenAI
from sklearn.linear_model import LinearRegression

# Extra import to call FastAPI
import requests

from setup_db import ChatHistory

logging.basicConfig(level=logging.INFO)

dotenv.load_dotenv()

# -----------------------------------------
# 1) Flask & DB Setup
# -----------------------------------------
server = Flask(__name__)
server.config['SECRET_KEY'] = 'your_secret_key'
server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(server)
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = 'login'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------
# 2) User Model
# -----------------------------------------
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)

with server.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

from datetime import datetime

class Dashboard(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# -----------------------------------------
# 3) Flask Routes
# -----------------------------------------
@server.route('/')
def home():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@server.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            return redirect(url_for('dashboard'))
        return render_template('login.html', message="Invalid username or password")
    return render_template('login.html')

@server.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return render_template('signup.html', message="Username already exists! Please login.")
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, password_hash=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))
    return render_template('signup.html')

@server.route('/logout')
@login_required
def logout():
    logout_user()
    session.clear()
    return redirect(url_for('login'))

@server.route('/dashboard')
@login_required
def dashboard():
    return redirect('/dash/')

# ✅ Initialize Dash inside Flask (restricted to logged-in users)
app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dash/',
    external_stylesheets=[dbc.themes.FLATLY]
)

@app.server.before_request
def restrict_dash_access():
    if request.path.startswith('/dash/') and not current_user.is_authenticated:
        return redirect(url_for('login'))

# ✅ Cache Setup
cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache'})

app.layout = html.Div(
    style={
        "margin": "0",
        "padding": "0",
        "backgroundColor": "#081028",
        "minHeight": "100vh",
        "fontFamily": "'Open Sans', sans-serif"
    },
    children=[
        dcc.Location(id='url', refresh=True),

        # ---------------- NAVBAR ----------------
        dbc.NavbarSimple(
            brand="AI Data Viz Dashboard",
            color="dark",
            dark=True,
            className="shadow mb-4",
            brand_style={"fontSize": "1.5rem", "fontWeight": "bold", "color": "#ffffff"},
            style={
                "background": "linear-gradient(to right, #8e44ad, #3498db)",
                "border": "none",
                "textAlign": "center",
                "width": "100%",
                "padding": "10px",
                "margin": "0px",
            },
            fluid=True,
            children=[
                dbc.NavItem(dbc.Button("Logout", id="logout-button", color="danger", className="ml-auto"))
            ]
        ),

        # ---------------- MAIN UPLOAD + FORM + SUMMARIES ----------------
        dbc.Row(
            style={"padding": "20px"},
            children=[
                # 1) CSV UPLOAD
                dbc.Col(
                    width=2,
                    children=[
                        html.H4("Step 1: Upload Your CSV File", style={"color": "#ffffff", "marginBottom": "15px"}),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div(
                                ['Drag & Drop or ', html.A('Select File', style={"color": "#8ac4ff"})],
                                style={'fontWeight': 'bold', "color": "#8ac4ff"}
                            ),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '2px',
                                'borderStyle': 'dashed',
                                'borderRadius': '10px',
                                'textAlign': 'center',
                                'backgroundColor': '#3a1c70',
                                'background': 'linear-gradient(to right, #852cb7, #0f1134)',
                                'color': '#dcd0ff',
                                'boxSizing': 'border-box',
                                'display': 'inline-block',
                                'margin': '10px 0',
                            },
                            multiple=False
                        ),
                        html.Div(id='output-data-upload', style={"marginBottom": "10px"}),

                        # Ingest CSV to Pinecone
                        dbc.Button("Ingest CSV to Pinecone", id="ingest-button", color="success", className="mt-2"),
                        html.Div(id="ingest-status", style={"color": "#dcd0f2", "marginTop": "10px"})
                    ]
                ),

                # 2) FORM: DATA EXPLORATION GOALS
                dbc.Col(
                    width=4,
                    children=[
                        html.H4("Step 2: Define Your Data Exploration Goals", style={"color": "#ffffff", "marginBottom": "15px"}),
                        dbc.Form([
                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Most important aspect to find in this dataset:", style={"color": "#dcd0ff"}),
                                    dbc.Input(
                                        id='data-aspect', type='text',
                                        placeholder="e.g., Trends over time, correlations",
                                        style={
                                            "backgroundColor": "#2c1e4e",
                                            "color": "#dcd0ff",
                                            "border": "1px solid #8ac4ff",
                                            "borderRadius": "5px",
                                            "padding": "10px",
                                            "fontSize": "14px"
                                        }
                                    ),
                                ], width=12),
                            ], className="mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Specific ranges of interest:", style={"color": "#dcd0ff"}),
                                    dbc.Input(
                                        id='range-interest', type='text',
                                        placeholder="e.g., Date range, value range",
                                        style={
                                            "backgroundColor": "#2c1e4e",
                                            "color": "#dcd0ff",
                                            "border": "1px solid #8ac4ff",
                                            "borderRadius": "5px",
                                            "padding": "10px",
                                            "fontSize": "14px"
                                        }
                                    ),
                                ], width=12),
                            ], className="mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Story you want to tell with this data:", style={"color": "#dcd0ff"}),
                                    dbc.Input(
                                        id='story-goal', type='text',
                                        placeholder="e.g., Sales growth over the last year",
                                        style={
                                            "backgroundColor": "#2c1e4e",
                                            "color": "#dcd0ff",
                                            "border": "1px solid #8ac4ff",
                                            "borderRadius": "5px",
                                            "padding": "10px",
                                            "fontSize": "14px"
                                        }
                                    ),
                                ], width=12),
                            ], className="mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Specific parameters to check in the frontend:", style={"color": "#dcd0ff"}),
                                    dbc.Input(
                                        id='param-check', type='text',
                                        placeholder="e.g., Certain columns or thresholds",
                                        style={
                                            "backgroundColor": "#2c1e4e",
                                            "color": "#dcd0ff",
                                            "border": "1px solid #8ac4ff",
                                            "borderRadius": "5px",
                                            "padding": "10px",
                                            "fontSize": "14px"
                                        }
                                    ),
                                ], width=12),
                            ], className="mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Prioritize the visualization goals:", style={"color": "#dcd0ff"}),
                                    dbc.Input(
                                        id='viz-priority', type='text',
                                        placeholder="e.g., 1. Correlation, 2. Trends",
                                        style={
                                            "backgroundColor": "#2c1e4e",
                                            "color": "#dcd0ff",
                                            "border": "1px solid #8ac4ff",
                                            "borderRadius": "5px",
                                            "padding": "10px",
                                            "fontSize": "14px"
                                        }
                                    ),
                                ], width=12),
                            ], className="mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Enter the sample size to visualize:", style={"color": "#dcd0ff"}),
                                    dbc.Input(
                                        id='sample-size', type='number',
                                        placeholder="e.g., 1000",
                                        style={
                                            "backgroundColor": "#2c1e4e",
                                            "color": "#dcd0ff",
                                            "border": "1px solid #8ac4ff",
                                            "borderRadius": "5px",
                                            "padding": "10px",
                                            "fontSize": "14px"
                                        }
                                    ),
                                ], width=12),
                            ], className="mb-3"),

                            dbc.Row([
                                dbc.Col([
                                    dbc.Label("Select a color scheme:", style={"color": "#ffffff"}),
                                    dcc.Dropdown(
                                        id='color-scheme',
                                        options=[
                                            {'label': 'Plotly', 'value': 'Plotly'},
                                            {'label': 'D3', 'value': 'D3'},
                                            {'label': 'G10', 'value': 'G10'},
                                            {'label': 'T10', 'value': 'T10'},
                                            {'label': 'Alphabet', 'value': 'Alphabet'},
                                            {'label': 'Dark24', 'value': 'Dark24'},
                                            {'label': 'Light24', 'value': 'Light24'},
                                            {'label': 'Set1', 'value': 'Set1'},
                                            {'label': 'Pastel', 'value': 'Pastel'},
                                            {'label': 'Viridis', 'value': 'Viridis'},
                                            {'label': 'Cividis', 'value': 'Cividis'},
                                            {'label': 'Inferno', 'value': 'Inferno'},
                                            {'label': 'Magma', 'value': 'Magma'},
                                            {'label': 'Plasma', 'value': 'Plasma'},
                                        ],
                                        value='Plotly',
                                        placeholder="Select a color scheme",
                                        style={
                                            'backgroundColor': '#3a1c70',
                                            'color': '#dcd0ff'
                                        }
                                    ),
                                ], width=12),
                            ], className="mb-3"),

                            dbc.Button(
                                "Submit",
                                id='submit-button',
                                color="primary",
                                className='w-100',
                                style={
                                    "background": "linear-gradient(to right, #8e44ad, #3498db)",
                                    "border": "none",
                                    "marginTop": "10px"
                                }
                            ),
                        ])
                    ]
                ),

                # 3) STATISTICAL SUMMARIES
                dbc.Col(
                    width=6,
                    children=[
                        html.H4("Statistical Summaries", style={"color": "#ffffff"}),
                        dcc.Loading(
                            id="loading-summary",
                            type="default",
                            children=html.Div(id='statistical-summary')
                        )
                    ]
                )
            ]
        ),

        html.Hr(style={"borderColor": "#dcd0ff"}),  # Purple line separator

        # ---------------- GENERATED VISUALIZATIONS ----------------
        dbc.Container(
            fluid=True,
            style={
                "background": "linear-gradient(to right, #8e44ad, #3498db)",
                "margin": "0",
                "padding": "20px",
                "borderRadius": "15px",
                "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.1)",
                "width": "100%"
            },
            children=[
                html.H4(
                    "Generated Visualizations",
                    style={
                        "color": "#ffffff",
                        "textAlign": "center",
                        "marginBottom": "20px"
                    }
                ),
                dbc.Container(
                    id='graphs-container',
                    fluid=True,
                    style={
                        "margin": "20px auto",
                        "padding": "20px",
                        "borderRadius": "15px",
                        "backgroundColor": "#081028",
                        "boxShadow": "0px 4px 6px rgba(0, 0, 0, 0.2)",
                        "width": "90%",
                        "color": "#ffffff"
                    }
                )
            ]
        ),

        html.Hr(style={"borderColor": "#dcd0f2"}),

        # ---------------- CHAT WITH CSV SECTION ----------------
        dbc.Container(
            fluid=True,
            style={
                "padding": "20px",
                "backgroundColor": "#081028"
            },
            children=[
                html.H3("Chat with Your CSV (Pinecone Data)", style={"color": "#ffffff", "marginBottom": "15px"}),
                dbc.Card(
                    style={"backgroundColor": "#2c1e4e", "padding": "15px", "border": "none"},
                    children=[
                        dbc.Row([
                            dbc.Col([
                                dcc.Input(
                                    id="chat-input",
                                    type="text",
                                    placeholder="Ask a question about the data...",
                                    style={
                                        "width": "100%",
                                        "backgroundColor": "#1d1b31",
                                        "color": "#ffffff",
                                        "border": "1px solid #8ac4ff",
                                        "borderRadius": "5px",
                                        "padding": "10px",
                                        "fontSize": "14px"
                                    }
                                )
                            ], width=8),
                            dbc.Col([
                                html.Button(
                                    "Ask Chatbot",
                                    id="chat-button",
                                    n_clicks=0,
                                    style={
                                        "background": "linear-gradient(to right, #8e44ad, #3498db)",
                                        "border": "none",
                                        "color": "#fff",
                                        "padding": "10px 20px",
                                        "borderRadius": "5px",
                                        "fontSize": "14px",
                                        "marginLeft": "10px",
                                        "cursor": "pointer"
                                    }
                                )
                            ], width=4, style={"textAlign": "right"})
                        ], justify="start"),

                        html.Div(
                            id="chat-response",
                            style={
                                "color": "#dcd0f2",
                                "marginTop": "20px",
                                "whiteSpace": "pre-wrap"
                            }
                        )
                    ]
                )
            ]
        ),

        html.Hr(style={"borderColor": "#dcd0f2"}),

        # ---------------- HISTORY SIDEBAR ----------------
        dbc.Container(
            fluid=True,
            style={
                "padding": "20px",
                "backgroundColor": "#081028"
            },
            children=[
                html.H4("Saved Dashboards", style={"color": "#ffffff", "textAlign": "center"}),
                html.Ul(id="saved-dashboards", style={"color": "#ffffff", "listStyleType": "none"}),
                html.H4("Past Chats", style={"color": "#ffffff", "textAlign": "center", "marginTop": "20px"}),
                html.Ul(id="past-chats", style={"color": "#ffffff", "listStyleType": "none"})
            ]
        )
    ]
)



# -----------------------------------------
# Utility / Helper Functions
# -----------------------------------------
def parse_contents(contents):
    logging.info("Parsing contents")
    if contents:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        encodings = ['utf-8','latin1','ISO-8859-1']
        for encoding in encodings:
            try:
                df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
                return df
            except UnicodeDecodeError:
                continue
    return None

def handle_missing_values(df):
    logging.info("Handling missing values")
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())
        else:
            df[column] = df[column].fillna(df[column].mode()[0])
    return df

def get_sample(df, sample_size):
    if sample_size and sample_size < len(df):
        return df.sample(n=sample_size)
    return df

# missing
def generate_statistical_cards(df):
    cards = []
    num_desc = df.describe(include=[np.number]).T.reset_index()
    num_desc.rename(columns={'index': 'column'}, inplace=True)

    for _, row in num_desc.iterrows():
        card = dbc.Card(
            [
                dbc.CardHeader(html.H5(row['column'], className='card-title')),
                dbc.CardBody(
                    [
                        html.P(f"Mean: {row['mean']:.2f}", className='card-text'),
                        html.P(f"Median: {row['50%']:.2f}", className='card-text'),
                        html.P(f"Std: {row['std']:.2f}", className='card-text'),
                        html.P(f"Min: {row['min']:.2f}", className='card-text'),
                        html.P(f"Max: {row['max']:.2f}", className='card-text'),
                    ]
                ),
            ],
            color="light",
            inverse=False,
            outline=True,
            style={'margin-bottom': '15px'}
        )
        cards.append(dbc.Col(card, width=3))

    return dbc.Row(cards)

# missing
def detect_relationships(df):
    relationships = []
    # Numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Correlation matrix for numerical variables
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        # Find pairs with high correlation
        high_corr = corr_matrix.unstack().reset_index()
        high_corr.columns = ['var1', 'var2', 'correlation']
        high_corr = high_corr[(high_corr['var1'] != high_corr['var2']) & (abs(high_corr['correlation']) > 0.7)]
        for _, row in high_corr.iterrows():
            relationships.append(f"Strong correlation ({row['correlation']:.2f}) between '{row['var1']}' and '{row['var2']}'.")
    else:
        relationships.append("Not enough numerical columns for correlation analysis.")

    # Chi-squared test for categorical variables
    for i in range(len(cat_cols)):
        for j in range(i+1, len(cat_cols)):
            cat_col1 = cat_cols[i]
            cat_col2 = cat_cols[j]
            try:
                contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                if p < 0.05:
                    relationships.append(f"Significant association between '{cat_col1}' and '{cat_col2}' (p={p:.4f}).")
            except Exception as e:
                logging.warning(f"Could not compute chi-squared test between '{cat_col1}' and '{cat_col2}': {e}")

    # ANOVA for numerical vs categorical variables
    for num_col in num_cols:
        for cat_col in cat_cols:
            try:
                groups = [group.dropna().values for name, group in df.groupby(cat_col)[num_col]]
                if len(groups) > 1 and all(len(group) > 1 for group in groups):
                    f_stat, p = f_oneway(*groups)
                    if p < 0.05:
                        relationships.append(f"Significant difference in '{num_col}' across groups of '{cat_col}' (p={p:.4f}).")
            except Exception as e:
                logging.warning(f"Could not compute ANOVA between '{num_col}' and '{cat_col}': {e}")

    return relationships



def save_dashboard_for_user(user_id, figures):
    """
    Save dashboard for a specific user via FastAPI
    """
    try:
        # Prepare dashboard data
        dashboard_data = {
            "user_id": user_id,
            "dashboard_name": f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            "figures": [fig.to_html(include_plotlyjs=False, full_html=False) for fig in figures]
        }

        # Send to FastAPI endpoint
        response = requests.post(
            "http://localhost:8000/save_dashboard", 
            json=dashboard_data
        )

        if response.status_code == 200:
            # Optionally log or handle successful save
            saved_dashboard = response.json()
            return saved_dashboard
        else:
            # Handle error
            print(f"Failed to save dashboard: {response.text}")
            return None
    except Exception as e:
        print(f"Error saving dashboard: {e}")
        return None

def fetch_user_dashboards(user_id):
    """
    Fetch saved dashboards for a specific user
    """
    try:
        response = requests.get(f"http://localhost:8000/get_user_dashboards/{user_id}")
        if response.status_code == 200:
            return response.json().get("dashboards", [])
        else:
            print(f"Failed to fetch dashboards: {response.text}")
            return []
    except Exception as e:
        print(f"Error fetching dashboards: {e}")
        return []

def load_user_dashboard(user_id, dashboard_name):
    """
    Load a specific dashboard for a user
    """
    try:
        response = requests.get(f"http://localhost:8000/load_user_dashboard/{user_id}/{dashboard_name}")
        if response.status_code == 200:
            # You might want to parse the HTML and recreate Plotly figures
            return response.text
        else:
            print(f"Failed to load dashboard: {response.text}")
            return None
    except Exception as e:
        print(f"Error loading dashboard: {e}")
        return None

# CHANGE THE CODE 

# missing
def generate_prompt(df, aspect, range_interest, story_goal, param_check, viz_priority):
    # Generate statistical summaries
    stats_summary = generate_statistical_summaries(df)
    # Detect relationships
    relationships = detect_relationships(df)
    relationships_text = '\n'.join(relationships)

    column_details = ", ".join([f"{col} ({dtype})" for col, dtype in zip(df.columns, df.dtypes)])
    prompt = f"""Given a dataset with columns {column_details}, here are some statistical summaries of the dataset:
{stats_summary}

Here are some detected relationships between columns:
{relationships_text}

I need recommendations for data cleaning and a diverse set of insightful visualizations that effectively showcase the data.
- Focus on the following key aspects: {aspect}.
- Consider these ranges of interest: {range_interest}.
- The story I want to tell is: {story_goal}.
- Please also check for these parameters: {param_check}.
- Prioritize the visualizations as follows: {viz_priority}.

First, thoroughly understand the schema by reviewing the column names, data types, and the statistical summaries provided. Determine which columns may require cleaning or preprocessing before visualization, such as handling missing values, outliers, or data type conversions.

Then, suggest a diverse set of visualization types that can provide valuable insights, such as:

- **Line charts** showing trends over time.
- **Scatter plots** to reveal correlations.
- **Bar charts** to compare categories.
- **Histograms** to understand distributions.
- **Heatmaps** to visualize correlations.
- **Box plots** to identify outliers.
- **Violin plots** for distribution shapes.
- **Area charts** for cumulative data.
- **Multi-line charts** for comparing multiple variables over time.
- **Combination charts** like bar and line together.

Please follow these specific guidelines for your response:
- Provide data cleaning steps one at a time.
- For each data cleaning step, specify the action and the columns it applies to.
- Use only these cleaning actions: remove_outliers, fill_missing_values, drop_missing_values, convert_to_datetime, remove_duplicates, normalize, standardize, filter_recent_years.
- Then, provide visualization types one at a time.
- For each visualization, specify the type and the best columns to use.
- Ensure each visualization is unique and offers different insights.
- Also, suggest any appropriate predictive models or forecasts that can be applied to the data, such as time series forecasting or regression models.
- For predictive models, provide the following format:
  - First line: ml_action = [action in lowercase]
  - Second line: columns = [list of column names]
- Use only these ML actions: time_series_forecasting, regression_analysis.
- Do not include any additional text, notes, or explanations in your response.
- Use only these types of graphs: line, scatter, bar, histogram, heatmap, box, violin, area, multi-line, combination.

Here is an example format for clarity:
clean_action = remove_outliers
columns = ['Sales']
viz_type = line
columns = ['Date', 'Sales']
ml_action = time_series_forecasting
columns = ['Date', 'Sales']

Ensure that your response maintains this format consistently throughout, as our parsing function depends on it.

Note: The dataset may contain historical data without recent dates. Please avoid suggesting cleaning steps that could remove all data, such as filtering by recent years, unless the data supports it.

Suggest data cleaning steps, visualizations, and predictive models that make the most sense based on the data schema and column relationships."""
    return prompt

# missing
def generate_statistical_summaries(df):
    summaries = []
    # Descriptive statistics for numerical columns
    num_desc = df.describe(include=[np.number]).T
    num_desc = num_desc[['mean', '50%', 'std', 'min', 'max']]
    num_desc.rename(columns={'50%': 'median'}, inplace=True)
    num_desc.reset_index(inplace=True)
    num_desc.rename(columns={'index': 'column'}, inplace=True)

    for _, row in num_desc.iterrows():
        summaries.append(
            f"- '{row['column']}': mean={row['mean']:.2f}, median={row['median']:.2f}, "
            f"std={row['std']:.2f}, min={row['min']:.2f}, max={row['max']:.2f}"
        )

    # Value counts for categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_columns:
        top_values = df[col].value_counts().head(3)
        value_summary = ', '.join([f"'{idx}': {val}" for idx, val in top_values.items()])
        summaries.append(f"- '{col}': Top values: {value_summary}")

    # Correlation matrix (top correlations)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        corr_pairs = upper_tri.unstack().dropna().reset_index()
        corr_pairs.columns = ['var1', 'var2', 'correlation']
        corr_pairs['abs_corr'] = corr_pairs['correlation'].abs()
        top_corr = corr_pairs.sort_values(by='abs_corr', ascending=False).head(3)
        for _, row in top_corr.iterrows():
            summaries.append(
                f"- Correlation between '{row['var1']}' and '{row['var2']}': {row['correlation']:.2f}"
            )
    else:
        summaries.append("- Not enough numeric columns for correlation analysis.")

    return '\n'.join(summaries)

# missing
@cache.memoize(timeout=300)
def query_openai(prompt):
    """ Send prompt to OpenAI and return the suggested visualization """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=600
    )
    logging.info(response.choices[0].message.content.strip())  # Debugging: See what OpenAI suggests
    return parse_ai_response(response.choices[0].message.content.strip())

# missing
def parse_ai_response(response):
    visualizations = []
    cleaning_steps = []
    ml_steps = []
    current_action_type = None  # 'clean_action', 'viz_type', or 'ml_action'
    current_action = None
    lines = response.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if "clean_action" in line:
            try:
                current_action_type = 'clean_action'
                current_action = line.split('=')[1].strip().strip("'").strip('"')
            except IndexError:
                logging.warning(f"Malformed clean_action line: {line}")
                current_action = None
        elif "viz_type" in line:
            try:
                current_action_type = 'viz_type'
                current_action = line.split('=')[1].strip().strip("'").strip('"')
            except IndexError:
                logging.warning(f"Malformed viz_type line: {line}")
                current_action = None
        elif "ml_action" in line:
            try:
                current_action_type = 'ml_action'
                current_action = line.split('=')[1].strip().strip("'").strip('"')
            except IndexError:
                logging.warning(f"Malformed ml_action line: {line}")
                current_action = None
        elif "columns" in line:
            try:
                columns_part = line.split('=')[1].strip()
                columns = eval(columns_part) if '[' in columns_part else [columns_part.strip("'").strip('"')]
                columns = [col.strip().strip("'").strip('"') for col in columns]
                if current_action_type == 'clean_action' and current_action and columns:
                    cleaning_steps.append({'action': current_action, 'columns': columns})
                    current_action = None
                elif current_action_type == 'viz_type' and current_action and columns:
                    visualizations.append({'type': current_action, 'columns': columns})
                    current_action = None
                elif current_action_type == 'ml_action' and current_action and columns:
                    ml_steps.append({'action': current_action, 'columns': columns})
                    current_action = None
            except (IndexError, SyntaxError) as e:
                logging.warning(f"Malformed columns line: {line}")
        else:
            logging.warning(f"Skipping unrecognized line: {line}")
    # Remove duplicate visualizations and cleaning steps
    cleaning_steps = deduplicate_cleaning_steps(cleaning_steps)
    visualizations = deduplicate_visualizations(visualizations)
    return cleaning_steps, visualizations, ml_steps

# missing
def deduplicate_visualizations(visualizations):
    seen = set()
    unique_visualizations = []
    for viz in visualizations:
        viz_repr = (viz['type'], tuple(viz['columns']))
        if viz_repr not in seen:
            seen.add(viz_repr)
            unique_visualizations.append(viz)
    return unique_visualizations

# missing
def deduplicate_cleaning_steps(cleaning_steps):
    seen = set()
    unique_steps = []
    for step in cleaning_steps:
        step_repr = (step['action'], tuple(step['columns']))
        if step_repr not in seen:
            seen.add(step_repr)
            unique_steps.append(step)
    return unique_steps

# missing
def apply_data_cleaning(df, cleaning_steps):
    action_mapping = {
        'convert_dates': 'convert_to_datetime',
        'handle_missing_values': 'fill_missing_values',
        'check_outliers': 'remove_outliers',
        'filter_data_by_years': 'filter_recent_years',
        'range_split_prices': 'bin_numerical_column',
        # Add more mappings if needed
    }
    for step in cleaning_steps:
        try:
            action = step['action']
            columns = step['columns']
            # Map action if necessary
            action = action_mapping.get(action, action)
            logging.info(f"Applying {action} on columns {columns}")
            df_before_step = df.copy()  # Keep a copy before applying the step
            rows_before = len(df)
            # Apply the cleaning action
            if action == 'remove_outliers':
                df = remove_outliers(df, columns)
            elif action == 'fill_missing_values':
                df = fill_missing_values(df, columns)
            elif action == 'drop_missing_values':
                df = drop_missing_values(df, columns)
            elif action == 'convert_to_datetime':
                df = convert_to_datetime(df, columns)
            elif action == 'remove_duplicates':
                df = df.drop_duplicates()
            elif action == 'normalize':
                df = normalize_columns(df, columns)
            elif action == 'standardize':
                df = standardize_columns(df, columns)
            elif action == 'filter_recent_years':
                df = filter_recent_years(df, columns)
            elif action == 'bin_numerical_column':
                df = bin_numerical_column(df, columns)
            else:
                logging.warning(f"Unknown cleaning action: {action}")
            rows_after = len(df)
            logging.info(f"Rows before: {rows_before}, Rows after: {rows_after}")
            if rows_after == 0:
                logging.warning(f"All data removed after applying {action} on columns {columns}. Reverting this step.")
                df = df_before_step.copy()
        except Exception as e:
            logging.error(f"Error applying {step['action']} to columns {step['columns']}: {e}")
    return df

def remove_outliers(df, columns):
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def fill_missing_values(df, columns):
    for col in columns:
        if col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])
    return df

def drop_missing_values(df, columns):
    df = df.dropna(subset=columns)
    return df

def convert_to_datetime(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Optionally drop rows where conversion failed
            df.dropna(subset=[col], inplace=True)
    return df

def normalize_columns(df, columns):
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val - min_val != 0:
                df[col] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[col] = 0
    return df

def standardize_columns(df, columns):
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            std = df[col].std()
            if std != 0:
                df[col] = (df[col] - df[col].mean()) / std
            else:
                df[col] = 0
    return df

def filter_recent_years(df, columns, years=5):
    for col in columns:
        if col in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            recent_year = pd.to_datetime('today').year - years  # Default to last 'years' years
            mask = df[col].dt.year >= recent_year
            mask &= df[col].notna()
            if mask.any():
                df = df.loc[mask]
                logging.info(f"Applied filter_recent_years on '{col}'. Rows remaining: {len(df)}.")
            else:
                logging.warning(f"No data within the last {years} years in column '{col}'. Skipping this filter.")
    return df

def bin_numerical_column(df, columns, bins=5):
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            df[col+'_binned'] = pd.qcut(df[col], q=bins, duplicates='drop')
    return df

def get_color_palette(color_scheme):
    # For qualitative color schemes
    qualitative_schemes = px.colors.qualitative.__dict__
    # For sequential color schemes
    sequential_schemes = px.colors.sequential.__dict__
    # For diverging color schemes
    diverging_schemes = px.colors.diverging.__dict__

    if color_scheme in qualitative_schemes:
        return qualitative_schemes[color_scheme]
    elif color_scheme in sequential_schemes:
        return sequential_schemes[color_scheme]
    elif color_scheme in diverging_schemes:
        return diverging_schemes[color_scheme]
    else:
        # Default
        return px.colors.qualitative.Plotly

def create_figure(df, viz_info, color_scheme):
    fig = None
    try:
        if viz_info and 'type' in viz_info and 'columns' in viz_info:
            columns_exist = all(col in df.columns for col in viz_info['columns'])
            if not columns_exist:
                logging.warning(f"One or more columns in {viz_info['columns']} do not exist in the dataframe.")
                return None

            # Ensure date columns are datetime
            for col in viz_info['columns']:
                if 'date' in col.lower():
                    df[col] = pd.to_datetime(df[col], errors='coerce')

            # Check if DataFrame is empty
            if df.empty:
                logging.warning("DataFrame is empty after cleaning. Cannot create figure.")
                return None

            # Get the color palette
            palette = get_color_palette(color_scheme)

            title = f"{viz_info['type'].capitalize()} of {' and '.join(viz_info['columns'])}"

            if viz_info['type'] == 'line' and len(viz_info['columns']) >= 2:
                df = df.sort_values(by=viz_info['columns'][0])
                fig = px.line(df, x=viz_info['columns'][0], y=viz_info['columns'][1],
                              color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None,
                              color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'scatter' and len(viz_info['columns']) >= 2:
                fig = px.scatter(df, x=viz_info['columns'][0], y=viz_info['columns'][1],
                                 color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None,
                                 color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'bar' and len(viz_info['columns']) >= 2:
                fig = px.bar(df, x=viz_info['columns'][0], y=viz_info['columns'][1],
                             color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None,
                             color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'histogram' and len(viz_info['columns']) >= 1:
                fig = px.histogram(df, x=viz_info['columns'][0],
                                   color=viz_info['columns'][1] if len(viz_info['columns']) > 1 else None,
                                   color_discrete_sequence=palette, nbins=50, title=title)
            elif viz_info['type'] == 'heatmap' and len(viz_info['columns']) >= 2:
                corr = df[viz_info['columns']].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale=palette, title='Correlation Heatmap')
            elif viz_info['type'] == 'box' and len(viz_info['columns']) >= 1:
                fig = px.box(df, y=viz_info['columns'][0],
                             x=viz_info['columns'][1] if len(viz_info['columns']) > 1 else None,
                             color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None,
                             color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'violin' and len(viz_info['columns']) >= 1:
                fig = px.violin(df, y=viz_info['columns'][0],
                                x=viz_info['columns'][1] if len(viz_info['columns']) > 1 else None,
                                color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None,
                                box=True, points='all', color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'area' and len(viz_info['columns']) >= 2:
                df = df.sort_values(by=viz_info['columns'][0])
                fig = px.area(df, x=viz_info['columns'][0], y=viz_info['columns'][1],
                              color=viz_info['columns'][2] if len(viz_info['columns']) > 2 else None,
                              color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'multi-line' and len(viz_info['columns']) >= 3:
                df = df.sort_values(by=viz_info['columns'][0])
                fig = px.line(df, x=viz_info['columns'][0], y=viz_info['columns'][1:],
                              color_discrete_sequence=palette, title=title)
            elif viz_info['type'] == 'combination' and len(viz_info['columns']) >= 2:
                # For simplicity, create a bar chart overlaid with a line chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=df[viz_info['columns'][0]],
                    y=df[viz_info['columns'][1]],
                    name=viz_info['columns'][1],
                    marker_color=palette[0]
                ))
                if len(viz_info['columns']) > 2 and viz_info['columns'][2] in df.columns:
                    fig.add_trace(go.Scatter(
                        x=df[viz_info['columns'][0]],
                        y=df[viz_info['columns'][2]],
                        name=viz_info['columns'][2],
                        marker_color=palette[1],
                        mode='lines+markers',
                        yaxis='y2'
                    ))
                    # Add a secondary y-axis
                    fig.update_layout(
                        yaxis=dict(
                            title=viz_info['columns'][1],
                        ),
                        yaxis2=dict(
                            title=viz_info['columns'][2],
                            overlaying='y',
                            side='right'
                        )
                    )
                fig.update_layout(title=title)
            else:
                logging.warning(f"Visualization type '{viz_info['type']}' is not supported.")
                return None

            if fig:
                fig.update_layout(
                    # Simplify the layout
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    # Remove gridlines
                    xaxis=dict(showgrid=False),
                    yaxis=dict(showgrid=False),
                    # Adjust font sizes and styles
                    title_font_size=16,
                    xaxis_title=viz_info['columns'][0],
                    yaxis_title=viz_info['columns'][1] if len(viz_info['columns']) > 1 else '',
                    legend_title_text=viz_info['columns'][2] if len(viz_info['columns']) > 2 else '',
                    font=dict(
                        family="Arial",
                        size=12,
                        color="Black"
                    ),
                    # Position the legend
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                # Remove chartjunk elements
                fig.update_traces(marker=dict(size=5), selector=dict(mode='markers'))
        else:
            logging.warning("Invalid visualization information provided.")
    except Exception as e:
        logging.error(f"Error in creating figure: {e}")
    return fig

def apply_ml_models(df, ml_steps):
    predictions = {}
    for step in ml_steps:
        action = step['action']
        columns = step['columns']
        if action == 'time_series_forecasting':
            prediction = time_series_forecasting(df, columns)
            if prediction is not None:
                predictions['time_series_forecasting'] = prediction
        elif action == 'regression_analysis':
            prediction = regression_analysis(df, columns)
            if prediction is not None:
                predictions['regression_analysis'] = prediction
        else:
            logging.warning(f"Unknown ML action: {action}")
    return predictions

def time_series_forecasting(df, columns):
    if len(columns) < 2:
        logging.warning("Not enough columns for time series forecasting.")
        return None
    date_col = columns[0]
    target_col = columns[1]
    if date_col not in df.columns or target_col not in df.columns:
        logging.warning(f"Columns {columns} not found in dataframe.")
        return None
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, target_col])
    df = df.sort_values(by=date_col)
    df['timestamp'] = df[date_col].map(pd.Timestamp.toordinal)
    X = df[['timestamp']]
    y = df[target_col]
    model = LinearRegression()
    model.fit(X, y)
    # Predict for the next 12 months
    last_date = df[date_col].max()
    future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), periods=12, freq='MS')
    future_timestamps = future_dates.map(pd.Timestamp.toordinal).values.reshape(-1,1)
    future_predictions = model.predict(future_timestamps)
    prediction_df = pd.DataFrame({date_col: future_dates, target_col: future_predictions})
    return prediction_df

def regression_analysis(df, columns):
    if len(columns) < 2:
        logging.warning("Not enough columns for regression analysis.")
        return None
    target_col = columns[0]
    feature_cols = columns[1:]
    missing_cols = [col for col in columns if col not in df.columns]
    if missing_cols:
        logging.warning(f"Columns {missing_cols} not found in dataframe.")
        return None
    df = df.dropna(subset=columns)
    X = df[feature_cols]
    y = df[target_col]
    # Encode categorical variables if any
    X = pd.get_dummies(X, drop_first=True)
    model = LinearRegression()
    model.fit(X, y)
    # For simplicity, return the coefficients
    coefficients = pd.Series(model.coef_, index=X.columns)
    intercept = model.intercept_
    result = {'coefficients': coefficients, 'intercept': intercept}
    return result

logging.basicConfig(level=logging.INFO)

# Example snippet for correlation tests:
from scipy.stats import chi2_contingency, f_oneway

def generate_statistical_cards(df):
    cards = []
    num_desc = df.describe(include=[np.number]).T.reset_index()
    num_desc.rename(columns={'index': 'column'}, inplace=True)
    for _, row in num_desc.iterrows():
        card = dbc.Card(
            [
                dbc.CardHeader(html.H5(row['column'], className='card-title')),
                dbc.CardBody(
                    [
                        html.P(f"Mean: {row['mean']:.2f}", className='card-text'),
                        html.P(f"Median: {row['50%']:.2f}", className='card-text'),
                        html.P(f"Std: {row['std']:.2f}", className='card-text'),
                        html.P(f"Min: {row['min']:.2f}", className='card-text'),
                        html.P(f"Max: {row['max']:.2f}", className='card-text'),
                    ]
                ),
            ],
            color="light",
            inverse=False,
            outline=True,
            style={'marginBottom': '15px'}
        )
        cards.append(dbc.Col(card, width=3))
    return dbc.Row(cards)

# More complex functions like detect_relationships, generate_prompt, query_openai, parse_ai_response, etc.
# Make sure they appear above your callback if used.

# ------------------------------------------------
# 5) Callback: Logout
# ------------------------------------------------
@app.callback(
    Output("url", "pathname"),
    Input("logout-button", "n_clicks"),
    prevent_initial_call=True
)
def logout_and_redirect(n_clicks):
    if n_clicks:
        logout_user()
        return "/login"

# ------------------------------------------------
# 6) Callback: Display "File Uploaded" text
# ------------------------------------------------
@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents')
)
def display_filename(contents):
    if contents:
        return html.Div("File uploaded successfully.", style={'color': '#dcd0f2','marginBottom':'10px'})
    return html.Div()

# ------------------------------------------------
# 7) Callback: Ingest CSV to Pinecone
# ------------------------------------------------
@app.callback(
    Output("ingest-status", "children"),
    [Input("ingest-button", "n_clicks")],
    [State("upload-data", "contents")]
)
def ingest_csv_to_pinecone(n_clicks, contents):
    """
    Call the FastAPI /upload_csv endpoint with base64 CSV.
    Improved error handling and user data isolation.
    """
    if not n_clicks:
        return ""
        
    if not contents:
        return "No CSV uploaded yet. Please upload a CSV file first."
        
    if not current_user.is_authenticated:
        return "Please log in to ingest data."
    
    try:
        # Set up the API endpoint
        fastapi_url = "http://127.0.0.1:8000/upload_csv"
        
        # Get user identification
        user_id = str(current_user.id)
        
        # Extract base64 data from the contents
        try:
            b64_data = contents.split(",")[1]
        except (IndexError, AttributeError) as e:
            logging.error(f"Error extracting base64 data: {e}")
            return "Error processing CSV data. Please try a different file."
        
        # Build the payload with user_id
        payload = {
            "user_id": user_id,
            "session_id": user_id,  # Include session_id for backward compatibility
            "csv_base64": b64_data
        }
        
        # Log the request (without the actual data for privacy)
        logging.info(f"Sending CSV ingestion request for user {user_id}")
        
        # Make the request with proper error handling
        try:
            response = requests.post(
                fastapi_url, 
                json=payload, 
                timeout=30
            )
            
            # Handle successful response
            if response.status_code == 200:
                try:
                    resp_data = response.json()
                    rows_ingested = resp_data.get('rows_ingested', 'unknown')
                    namespace = resp_data.get('namespace', user_id)
                    
                    # Create a success message with helpful information
                    success_message = dbc.Alert(
                        [
                            html.H4("CSV Ingestion Successful", className="alert-heading"),
                            html.P(f"Rows ingested: {rows_ingested}"),
                            html.P(f"Your data is ready for querying in the chat section below.")
                        ],
                        color="success",
                        dismissable=True
                    )
                    
                    logging.info(f"Successful ingestion for user {user_id}: {rows_ingested} rows")
                    return success_message
                    
                except ValueError as json_err:
                    logging.error(f"Error parsing JSON response: {json_err}")
                    return f"Server returned invalid data. Please try again later."
            
            # Handle error responses
            else:
                try:
                    error_detail = response.json().get('detail', response.text)
                except:
                    error_detail = response.text or f"Status code: {response.status_code}"
                
                error_message = dbc.Alert(
                    [
                        html.H4("CSV Ingestion Failed", className="alert-heading"),
                        html.P(f"Error: {error_detail}"),
                        html.P("Please check your CSV file and try again.")
                    ],
                    color="danger",
                    dismissable=True
                )
                
                logging.error(f"CSV ingestion failed for user {user_id}: {error_detail}")
                return error_message
                
        except requests.exceptions.Timeout:
            logging.error(f"Timeout during CSV ingestion for user {user_id}")
            return "The request timed out. Your CSV may be too large or the server is busy."
            
        except requests.exceptions.ConnectionError:
            logging.error(f"Connection error during CSV ingestion for user {user_id}")
            return "Could not connect to the server. Please try again later."
            
        except Exception as req_err:
            logging.error(f"Request error during CSV ingestion: {req_err}")
            return f"Request failed: {str(req_err)}"
            
    except Exception as e:
        logging.error(f"Unexpected error in CSV ingestion: {e}")
        return f"An unexpected error occurred: {str(e)}"

# ------------------------------------------------
# 8) Callback: Generate Summaries & Visuals
# ------------------------------------------------
# Provide or import your "detect_relationships", "generate_prompt", "query_openai", etc. here as well.

@app.callback(
    [Output('statistical-summary', 'children'),
     Output('graphs-container', 'children')],
    Input('submit-button', 'n_clicks'),
    State('upload-data', 'contents'),
    State('data-aspect', 'value'),
    State('range-interest', 'value'),
    State('story-goal', 'value'),
    State('param-check', 'value'),
    State('viz-priority', 'value'),
    State('sample-size', 'value'),
    State('color-scheme', 'value')
)
def update_graph(n_clicks, contents, aspect, range_interest, story_goal, param_check, viz_priority, sample_size, color_scheme):
    logging.info("Update Graph Callback triggered")
    if contents and n_clicks:
        df = parse_contents(contents)
        if df is not None:
            # Possibly handle_missing_values(df) if you want
            df_original = df.copy()
            df = get_sample(df, sample_size)

            # 1) Summaries
            summary_cards = generate_statistical_cards(df_original)

            # 2) Use GPT to suggest cleaning steps & visuals
            prompt = generate_prompt(df, aspect, range_interest, story_goal, param_check, viz_priority)
            cleaning_steps, visualization_suggestions, ml_steps = query_openai(prompt)
            df = apply_data_cleaning(df, cleaning_steps)
            if df.empty:
                logging.warning("DataFrame empty after cleaning. No visuals to show.")
                return summary_cards, [html.Div("No data available after cleaning.", style={'color': 'red'})]

            # 3) ML models
            predictions = apply_ml_models(df, ml_steps)

            # 4) Build figures
            figures = []
            for i, viz in enumerate(visualization_suggestions):
                logging.info(f"Creating figure {i+1}")
                fig = create_figure(df, viz, color_scheme)
                if fig is not None:
                    figures.append(dcc.Graph(figure=fig, id=f'graph-output{i+1}'))
                else:
                    logging.warning(f"Figure {i+1} could not be created.")

            # 5) If ML predictions exist, visualize them
            if 'time_series_forecasting' in predictions:
                pred_df = predictions['time_series_forecasting']
                date_col = pred_df.columns[0]
                target_col = pred_df.columns[1]
                fig = px.line(pred_df, x=date_col, y=target_col,
                              title='Time Series Forecast',
                              color_discrete_sequence=get_color_palette(color_scheme))
                figures.append(dcc.Graph(figure=fig, id='ml-graph-1'))

            if 'regression_analysis' in predictions:
                result = predictions['regression_analysis']
                coeffs = result['coefficients'].reset_index()
                coeffs.columns = ['Feature', 'Coefficient']
                fig = px.bar(coeffs, x='Feature', y='Coefficient',
                             title='Regression Coefficients',
                             color_discrete_sequence=get_color_palette(color_scheme))
                figures.append(dcc.Graph(figure=fig, id='ml-graph-2'))

            # 6) Layout figures in rows
            rows = []
            for i in range(0, len(figures), 2):
                row = dbc.Row([
                    dbc.Col(figures[i], width=6) if i < len(figures) else None,
                    dbc.Col(figures[i+1], width=6) if i+1 < len(figures) else None
                ], style={'marginBottom': '20px'})
                rows.append(row)

            # 7) Save dashboard via FastAPI endpoint
            try:
                # Extract Plotly figures from the graphs
                plotly_figures = [graph.figure for graph in figures if hasattr(graph, 'figure')]
                
                if not plotly_figures:
                    logging.warning("No valid figures to save in the dashboard")
                else:
                    try:
                        # Save locally first as a backup
                        file_path, file_name = save_dashboard_html(plotly_figures, current_user.id)
                        
                        # Create a new dashboard record in the database
                        new_dashboard = Dashboard(
                            user_id=current_user.id,
                            file_path=file_path,
                            dashboard_name=f"Dashboard {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                        )
                        db.session.add(new_dashboard)
                        db.session.commit()
                        logging.info(f"Dashboard saved locally at {file_path} and recorded in database")
                        
                        # Also try to save via FastAPI as a secondary method
                        dashboard_data = {
                            "user_id": str(current_user.id),
                            "dashboard_name": file_name,
                            "figures": [
                                fig.to_html(include_plotlyjs=False, full_html=False) 
                                for fig in plotly_figures
                            ]
                        }
                        
                        try:
                            response = requests.post(
                                "http://localhost:8000/save_dashboard", 
                                json=dashboard_data,
                                timeout=10  # Add timeout to prevent hanging
                            )
                            
                            if response.status_code == 200:
                                saved_dashboard = response.json()
                                logging.info(f"Dashboard also saved via API: {saved_dashboard}")
                            else:
                                logging.warning(f"FastAPI dashboard save failed: {response.text}")
                        except requests.exceptions.RequestException as api_err:
                            # If API fails, we already have local backup
                            logging.warning(f"FastAPI dashboard save request failed: {api_err}")
                    
                    except Exception as save_err:
                        logging.error(f"Error saving dashboard: {save_err}")
                        # Still return the visualizations even if saving fails
            
            except Exception as e:
                logging.error(f"Error in dashboard saving process: {e}")
                # Continue to return visualizations even if saving fails

    return summary_cards, rows

# Modify save_dashboard_html function to be more flexible
def save_dashboard_html(figures, file_path):
    """
    Combines a list of Plotly figures into one HTML page and saves it.
    More robust version that handles different figure types.
    """
    try:
        html_parts = [
            "<html>",
            "<head>",
            "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>",
            "<style>",
            "body { margin: 0; padding: 0; font-family: Arial, sans-serif; }",
            ".dashboard { display: flex; flex-wrap: wrap; justify-content: center; }",
            ".graph-container { width: 50%; padding: 10px; box-sizing: border-box; }",
            "</style>",
            "</head>",
            "<body>",
            "<div class='dashboard'>"
        ]

        for fig in figures:
            # Check if figure is a Plotly figure or already an HTML string
            if hasattr(fig, 'to_html'):
                html_snippet = fig.to_html(include_plotlyjs=False, full_html=False)
            elif isinstance(fig, str):
                html_snippet = fig
            else:
                logging.warning(f"Unsupported figure type: {type(fig)}")
                continue

            # Wrap each figure in a container div
            html_parts.append(f"<div class='graph-container'>{html_snippet}</div>")

        html_parts.extend([
            "</div>",
            "</body>",
            "</html>"
        ])

        html_str = "\n".join(html_parts)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_str)
        
        logging.info(f"Dashboard saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving dashboard HTML: {e}")


# ------------------------------------------------
# 9) Callback: Ask Chatbot
# ------------------------------------------------
@app.callback(
    Output("chat-response", "children"),
    [Input("chat-button", "n_clicks")],
    [State("chat-input", "value")]
)
def ask_chatbot(n_clicks, query):
    if n_clicks and query:
        fastapi_ask_url = "http://127.0.0.1:8000/ask"
        
        # Prepare payload with user_id instead of session_id
        payload = {
            "input": query,
            "user_id": str(current_user.id),  # Ensure user_id is a string
            "session_id": str(current_user.id),  # Optional, but can be kept
        }
        
        try:
            # Use requests.post with streaming
            r = requests.post(fastapi_ask_url, json=payload, stream=True, timeout=30)
            
            # Check for successful response
            if r.status_code != 200:
                # Try to get more detailed error information
                error_details = r.text
                return f"Error from chatbot server: {r.status_code} - {error_details}"

            # Collect streaming response
            full_text = ""
            response_chunks = []
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    decoded_chunk = chunk.decode("utf-8", errors="ignore")
                    full_text += decoded_chunk
                    response_chunks.append(decoded_chunk)

            # Save the chat history to the database
            try:
                new_chat = ChatHistory(
                    user_id=current_user.id, 
                    query=query, 
                    response=full_text
                )
                db.session.add(new_chat)
                db.session.commit()
            except Exception as db_error:
                print(f"Error saving chat history: {db_error}")
                # Optionally log the error or handle it as needed

            return full_text

        except requests.exceptions.RequestException as req_error:
            # Handle specific request-related exceptions
            error_message = f"Request failed: {str(req_error)}"
            print(error_message)  # Log the error
            return error_message
        
        except Exception as e:
            # Catch-all for any other unexpected errors
            error_message = f"Unexpected error: {str(e)}"
            print(error_message)  # Log the error
            return error_message

    return ""


def fetch_saved_dashboards():
    """Fetch the list of saved dashboards from FastAPI."""
    response = requests.get("http://localhost:8000/get_saved_dashboards")
    if response.status_code == 200:
        return response.json().get("dashboards", [])
    return []

# ------------------------------------------------
# 10) Callback: Display Saved Dashboards for Current User
# ------------------------------------------------
@app.callback(
    Output("saved-dashboards", "children"),
    [Input("refresh-history-button", "n_clicks"),
     Input('interval-component', 'n_intervals')]
)
def display_saved_dashboards(n_clicks, n_intervals):
    """
    Display the user's saved dashboards as clickable list items.
    Refresh when button is clicked or at intervals.
    """
    try:
        if not current_user.is_authenticated:
            return html.Li("Please log in to view saved dashboards", style={"color": "#ff9999"})

        # Get dashboards from the database
        user_dashboards = Dashboard.query.filter_by(user_id=current_user.id).order_by(Dashboard.created_at.desc()).all()
        
        if not user_dashboards:
            return html.Li("No saved dashboards found", style={"color": "#aaaaaa"})
        
        dashboard_items = []
        
        for dashboard in user_dashboards:
            # Create path for viewing the dashboard
            file_path = dashboard.file_path
            # Ensure the file exists
            if os.path.exists(file_path):
                # Create URL that will serve the dashboard HTML
                dashboard_url = f"/view_dashboard/{dashboard.id}"
                dashboard_name = dashboard.dashboard_name or f"Dashboard {dashboard.created_at.strftime('%Y-%m-%d %H:%M')}"
                
                item = html.Li([
                    html.A(
                        dashboard_name,
                        href=dashboard_url,
                        target="_blank",
                        style={
                            "color": "#8ac4ff",
                            "textDecoration": "none",
                            "display": "block",
                            "padding": "8px 15px",
                            "margin": "5px 0",
                            "borderRadius": "5px",
                            "background": "linear-gradient(to right, rgba(60, 20, 120, 0.8), rgba(30, 10, 60, 0.8))",
                            "transition": "background 0.3s"
                        }
                    ),
                    html.Small(
                        f"Created: {dashboard.created_at.strftime('%Y-%m-%d %H:%M')}",
                        style={"color": "#aaaaaa", "marginLeft": "15px"}
                    )
                ], style={"marginBottom": "10px"})
                
                dashboard_items.append(item)
            else:
                # File doesn't exist, but entry is in DB - log this discrepancy
                logging.warning(f"Dashboard file not found: {file_path} for dashboard ID {dashboard.id}")
                # Could optionally delete the DB entry here
        
        return dashboard_items
        
    except Exception as e:
        logging.error(f"Error displaying saved dashboards: {e}")
        return html.Li(f"Error loading dashboards: {str(e)}", style={"color": "#ff9999"})

# Add route to serve saved dashboard files
@server.route('/view_dashboard/<int:dashboard_id>')
@login_required
def view_dashboard(dashboard_id):
    try:
        # Get the dashboard from the database
        dashboard = Dashboard.query.filter_by(id=dashboard_id, user_id=current_user.id).first()
        
        if not dashboard:
            return "Dashboard not found or you don't have permission to view it", 404
        
        # Check if the file exists
        if not os.path.exists(dashboard.file_path):
            return "Dashboard file not found", 404
        
        # Serve the HTML file
        return send_file(dashboard.file_path)
    except Exception as e:
        logging.error(f"Error serving dashboard {dashboard_id}: {e}")
        return f"Error: {str(e)}", 500

# ------------------------------------------------
# 10) Main
# ------------------------------------------------
if __name__ == '__main__':
    server.run(debug=True, port=5000)
