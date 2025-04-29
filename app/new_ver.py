# """
# dash_flask_app.py

# Flask+Dash app:
# - Auth (login/signup)
# - CSV Upload + local analysis
# - Button to Ingest CSV to Pinecone (calls /upload_csv)
# - Chat input to ask GPT-based questions about the ingested data (calls /ask)
# """

import os
import logging
import base64
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dotenv
import uuid
from datetime import datetime, timedelta
import requests
import sqlite3
from sqlalchemy import text
import warnings
import json
from jinja2 import Template
from flask import render_template_string

# Initialize database if it doesn't exist
db_path = 'users.db'
try:
    # Only create tables if database doesn't exist
    if not os.path.exists(db_path):
        print(f"Creating new database at: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create user table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS user (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        ''')
        
        # Create dashboard table with all required columns
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS dashboard (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_path TEXT NOT NULL,
            dashboard_name TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user(id)
        )
        ''')
        
        # Create chat_history table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            query TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user(id)
        )
        ''')
        
        conn.commit()
        conn.close()
        print("Database tables created successfully")
    else:
        print(f"Using existing database at: {db_path}")
except Exception as e:
    print(f"Error initializing database: {e}")

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State
from dash.exceptions import PreventUpdate

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
from fastapi_chatbot import mongo_db

from flask import Flask, redirect, url_for, session, request
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
from flask_login import current_user, login_required
from flask_caching import Cache

from flask_cors import CORS
from cors_config import configure_cors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dotenv.load_dotenv()

# -----------------------------------------
# 1) Flask & DB Setup
# -----------------------------------------
server = Flask(__name__)
server = configure_cors(server)  # Configure CORS with our settings
server.config['SECRET_KEY'] = 'your_secret_key'
server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
server.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(server)
login_manager = LoginManager()
login_manager.init_app(server)
login_manager.login_view = 'login'

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Add this function to create or migrate the database as needed
def migrate_database():
    try:
        # Connect to SQLite database (will create if doesn't exist)
        conn = sqlite3.connect('users.db')  # Match the SQLAlchemy database name
        cursor = conn.cursor()
        
        # Check if dashboard table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='dashboard'")
        table_exists = cursor.fetchone()
        
        if not table_exists:
            # Create dashboard table if it doesn't exist
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS dashboard (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                dashboard_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES user(id)
            )
            ''')
            print("Created dashboard table")
        
        # Check if dashboard_name column exists
        cursor.execute("PRAGMA table_info(dashboard)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # Add dashboard_name column if it doesn't exist
        if 'dashboard_name' not in column_names:
            print("Adding dashboard_name column to dashboard table")
            cursor.execute("ALTER TABLE dashboard ADD COLUMN dashboard_name TEXT DEFAULT 'Unnamed Dashboard'")
        
        conn.commit()
        conn.close()
        print("Database migration completed successfully")
        
    except Exception as e:
        print(f"Database migration error: {str(e)}")

# Call migrate_database instead of recreate_dashboard_table
with server.app_context():
    migrate_database()

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
    """
    Load user from the database using the modern SQLAlchemy approach.
    """
    try:
        with db.session.begin():
            return db.session.get(User, int(user_id))
    except Exception as e:
        logging.error(f"Error loading user: {e}")
        return None

from datetime import datetime

class Dashboard(db.Model):
    __tablename__ = 'dashboard'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    file_path = db.Column(db.String(256), nullable=False)
    dashboard_name = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship with User
    user = db.relationship('User', backref=db.backref('dashboards', lazy=True))

class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship with User
    user = db.relationship('User', backref=db.backref('chat_history', lazy=True))

# Create the saved_dashboards directory if it doesn't exist
DASHBOARD_FOLDER = "saved_dashboards"
os.makedirs(DASHBOARD_FOLDER, exist_ok=True)

# -----------------------------------------
# Dashboard Functions
# -----------------------------------------
def save_dashboard_html(figures, file_path):
    """Save a list of plotly figures as an HTML dashboard using proper Plotly figure saving."""
    try:
        import plotly.graph_objects as go
        from jinja2 import Template
        
        # Create a template for the dashboard
        template = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Dashboard</title>
                <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
                <style>
                    body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background-color: #081028; color: white; }
                    .dashboard { display: flex; flex-wrap: wrap; justify-content: center; }
                    .graph-container { width: 90%; max-width: 800px; margin: 15px auto; padding: 10px; background-color: #1d1b31; border-radius: 8px; }
                    h1 { text-align: center; color: #fff; }
                </style>
            </head>
            <body>
                <h1>Data Visualization Dashboard</h1>
                <div class="dashboard">
                    {% for fig in figures %}
                    <div class="graph-container">
                        {{ fig }}
                    </div>
                    {% endfor %}
                </div>
            </body>
        </html>
        """
        
        # Convert figure dictionaries to Plotly figures and generate HTML
        figure_htmls = []
        for fig_dict in figures:
            if isinstance(fig_dict, dict):
                # Create a Plotly figure from the dictionary
                fig = go.Figure(fig_dict)
                # Generate HTML for the figure
                figure_htmls.append(fig.to_html(full_html=False, include_plotlyjs=False))
            else:
                # If it's already a Plotly figure
                figure_htmls.append(fig_dict.to_html(full_html=False, include_plotlyjs=False))
        
        # Render the template with the figures
        j2_template = Template(template)
        html_content = j2_template.render(figures=figure_htmls)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return True
    except Exception as e:
        logging.error(f"Error saving dashboard: {str(e)}")
        return False

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

@server.route('/view_dashboard/<int:dashboard_id>', methods=['GET'])
@login_required
def view_dashboard(dashboard_id):
    """
    View a saved dashboard by ID
    """
    try:
        logger.info(f"Attempting to view dashboard {dashboard_id} for user {current_user.id}")
        
        # Get the dashboard from the database
        dashboard = Dashboard.query.filter_by(
            id=dashboard_id,
            user_id=current_user.id
        ).first()
        
        if not dashboard:
            logger.error(f"Dashboard {dashboard_id} not found for user {current_user.id}")
            return render_template_string("""
                <h1>Dashboard Not Found</h1>
                <p>The requested dashboard does not exist or you don't have permission to view it.</p>
                <p>Current user ID: {{ user_id }}</p>
                <p>Requested dashboard ID: {{ dashboard_id }}</p>
            """, user_id=current_user.id, dashboard_id=dashboard_id), 404
            
        file_path = dashboard.file_path
        logger.info(f"Found dashboard with path: {file_path}")
        
        # Check if the file exists
        if not os.path.exists(file_path):
            # Try to find the file in the user's dashboard directory
            base_dir = os.path.join(os.getcwd(), 'app', 'dashboards')
            user_dir = os.path.join(base_dir, str(current_user.id))
            possible_file = os.path.join(user_dir, os.path.basename(file_path))
            
            if os.path.exists(possible_file):
                file_path = possible_file
                logger.info(f"Found file at alternative path: {file_path}")
            else:
                logger.error(f"File not found at either {file_path} or {possible_file}")
                return render_template_string("""
                    <h1>Dashboard File Not Found</h1>
                    <p>The dashboard file could not be found at the expected location.</p>
                    <p>Expected path: {{ file_path }}</p>
                    <p>Alternative path: {{ alt_path }}</p>
                """, file_path=file_path, alt_path=possible_file), 404
        
        # Serve the file
        logger.info(f"Serving dashboard file: {file_path}")
        return send_file(file_path)
        
    except Exception as e:
        logger.error(f"Error serving dashboard {dashboard_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return render_template_string("""
            <h1>Error Loading Dashboard</h1>
            <p>An error occurred while trying to load the dashboard.</p>
            <p>Error: {{ error }}</p>
        """, error=str(e)), 500

@server.route('/check_dashboards', methods=['GET'])
@login_required
def check_dashboards():
    """Route to check dashboard table contents"""
    try:
        # First check if user is authenticated
        if not current_user.is_authenticated:
            return redirect(url_for('login'))
            
        # Get current user's dashboards
        user_dashboards = Dashboard.query.filter_by(user_id=current_user.id).all()
        
        # Get all dashboards (for admin view)
        all_dashboards = Dashboard.query.all()
        
        # Create HTML template
        template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dashboard Check</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>Dashboard Check</h1>
            
            <h2>Your Dashboards (User ID: {{ current_user.id }})</h2>
            {% if user_dashboards %}
            <table>
                <tr>
                    <th>ID</th>
                    <th>Name</th>
                    <th>File Path</th>
                    <th>Created At</th>
                </tr>
                {% for dash in user_dashboards %}
                <tr>
                    <td>{{ dash.id }}</td>
                    <td>{{ dash.dashboard_name }}</td>
                    <td>{{ dash.file_path }}</td>
                    <td>{{ dash.created_at }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No dashboards found for your account.</p>
            {% endif %}
            
            <h2>All Dashboards in Database</h2>
            {% if all_dashboards %}
            <table>
                <tr>
                    <th>ID</th>
                    <th>User ID</th>
                    <th>Name</th>
                    <th>File Path</th>
                    <th>Created At</th>
                </tr>
                {% for dash in all_dashboards %}
                <tr>
                    <td>{{ dash.id }}</td>
                    <td>{{ dash.user_id }}</td>
                    <td>{{ dash.dashboard_name }}</td>
                    <td>{{ dash.file_path }}</td>
                    <td>{{ dash.created_at }}</td>
                </tr>
                {% endfor %}
            </table>
            {% else %}
            <p>No dashboards found in the database.</p>
            {% endif %}
        </body>
        </html>
        """
        
        return render_template_string(
            template,
            current_user=current_user,
            user_dashboards=user_dashboards,
            all_dashboards=all_dashboards
        )
            
    except Exception as e:
        logging.error(f"Error checking dashboard table: {e}")
        return f"Error checking dashboard table: {str(e)}", 500

@server.route('/dash/load_dashboard/<dashboard_name>')
def serve_dashboard(dashboard_name):
    if not current_user.is_authenticated:
        return redirect(url_for('login'))
    
    try:
        # Load the dashboard content
        dashboard_content = load_user_dashboard(str(current_user.id), dashboard_name)
        
        if dashboard_content:
            return dashboard_content
        else:
            return "Dashboard not found or could not be loaded", 404
    except Exception as e:
        print(f"Error serving dashboard: {e}")
        return f"Error loading dashboard: {str(e)}", 500

@server.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

# -----------------------------------------
# Main app setup and initialization 
# -----------------------------------------

# Create all tables first
with server.app_context():
    # Check if tables exist before creating
    inspector = db.inspect(db.engine)
    existing_tables = inspector.get_table_names()
    
    if not all(table in existing_tables for table in ['user', 'dashboard', 'chat_history']):
        print("Creating missing database tables...")
        db.create_all()
        print("Database tables created successfully")
    else:
        print("All required tables already exist")

# Initialize Dash with Flask server
app = dash.Dash(
    __name__,
    server=server,
    routes_pathname_prefix='/dash/',
    external_stylesheets=[dbc.themes.DARKLY],
    title="Data Insights Dashboard"
)

app.config.suppress_callback_exceptions = True
app.scripts.config.serve_locally = True
app.css.config.serve_locally = True

# Set up caching
cache = Cache(app.server, config={
    'CACHE_TYPE': 'simple',
    'CACHE_DEFAULT_TIMEOUT': 300
})

@app.server.before_request
def restrict_dash_access():
    print(f"DEBUG: before_request - path={request.path}, authenticated={current_user.is_authenticated if hasattr(current_user, 'is_authenticated') else 'Unknown'}")
    if request.path.startswith('/dash/') and not current_user.is_authenticated:
        return redirect(url_for('login'))

# ✅ Cache Setup
cache = Cache(app.server, config={'CACHE_TYPE': 'SimpleCache'})

print("DEBUG: Creating Dash app layout")
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
        
        # Add a hidden div to store the current user ID
        html.Div(id='current-user-store', style={'display': 'none'}, children="0"),

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

                            # Button Div with both Submit and Save buttons
                            dbc.Row([
                                dbc.Col([
                                    dbc.Button(
                                        "Generate Visualizations", 
                                        id="submit-button", 
                                        color="primary", 
                                        className="mt-3 w-100",
                                        style={
                                            "fontWeight": "bold",
                                            "backgroundColor": "#8e44ad",
                                            "borderColor": "#7d3c98",
                                            "boxShadow": "0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08)"
                                        }
                                    ),
                                ], width=6),
                                dbc.Col([
                                    dbc.Button(
                                        "Save Dashboard", 
                                        id="save-dashboard-button", 
                                        color="success", 
                                        className="mt-3 w-100",
                                        style={
                                            "fontWeight": "bold",
                                            "backgroundColor": "#2ecc71",
                                            "borderColor": "#27ae60",
                                            "boxShadow": "0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08)"
                                        }
                                    ),
                                ], width=6),
                            ]),
                            
                            # Status message for save operation
                            html.Div(id="save-status", style={"marginTop": "10px", "color": "#ffffff"}),
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
                
                # Add refresh controls for chat history:
                html.Button(
                    "Refresh Chat History",
                    id="refresh-history-button",
                    n_clicks=0,
                    style={
                        "background": "linear-gradient(to right, #8e44ad, #3498db)",
                        "border": "none",
                        "color": "#fff",
                        "padding": "10px 20px",
                        "borderRadius": "20px",  # Increased border radius for a rounder look
                        "fontSize": "14px",
                        "cursor": "pointer",
                        "marginBottom": "10px"
                    }
                ),
                dcc.Interval(id="interval-component", interval=30000, n_intervals=0, max_intervals=5),
                
                # Display past chats
                html.Ul(id="past-chats", style={"color": "#ffffff", "listStyleType": "none", "overflowY": "scroll", "maxHeight": "400px"})
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
    import warnings
    
    # Numerical columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    # Categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    # Correlation matrix for numerical variables
    if len(num_cols) >= 2:
        try:
            corr_matrix = df[num_cols].corr()
            # Find pairs with high correlation
            high_corr = corr_matrix.unstack().reset_index()
            high_corr.columns = ['var1', 'var2', 'correlation']
            high_corr = high_corr[(high_corr['var1'] != high_corr['var2']) & (abs(high_corr['correlation']) > 0.7)]
            for _, row in high_corr.iterrows():
                relationships.append(f"Strong correlation ({row['correlation']:.2f}) between '{row['var1']}' and '{row['var2']}'.")
        except Exception as e:
            logging.warning(f"Error computing correlations: {e}")
    else:
        relationships.append("Not enough numerical columns for correlation analysis.")

    # Chi-squared test for categorical variables
    if len(cat_cols) >= 2:
        for i in range(len(cat_cols)):
            for j in range(i+1, len(cat_cols)):
                cat_col1 = cat_cols[i]
                cat_col2 = cat_cols[j]
                try:
                    contingency_table = pd.crosstab(df[cat_col1], df[cat_col2])
                    # Skip if table is too small
                    if contingency_table.shape[0] < 2 or contingency_table.shape[1] < 2:
                        continue
                    
                    # Skip if too many missing values
                    expected_values = chi2_contingency(contingency_table)[3]
                    if (expected_values < 5).any():
                        continue
                        
                    chi2, p, dof, expected = chi2_contingency(contingency_table)
                    if p < 0.05:
                        relationships.append(f"Significant association between '{cat_col1}' and '{cat_col2}' (p={p:.4f}).")
                except Exception as e:
                    logging.warning(f"Could not compute chi-squared test between '{cat_col1}' and '{cat_col2}': {e}")

    # ANOVA for numerical vs categorical variables
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for num_col in num_cols:
            for cat_col in cat_cols:
                try:
                    # Check if the numerical column has variance
                    if df[num_col].std() == 0:
                        continue
                        
                    # Check if we have enough groups with enough samples
                    groups = [group.dropna().values for name, group in df.groupby(cat_col)[num_col]]
                    valid_groups = [g for g in groups if len(g) > 1 and g.std() > 0]
                    
                    if len(valid_groups) > 1:
                        f_stat, p = f_oneway(*valid_groups)
                        if not np.isnan(p) and p < 0.05:
                            relationships.append(f"Significant difference in '{num_col}' across groups of '{cat_col}' (p={p:.4f}).")
                except Exception as e:
                    logging.warning(f"Could not compute ANOVA between '{num_col}' and '{cat_col}': {e}")

    return relationships



# Consolidated Dashboard Functions
def save_dashboard_for_user(user_id, figures):
    """
    Save dashboard figures for a specific user with improved error handling
    """
    try:
        logger.info(f"Saving dashboard for user {user_id}")
        
        # Create user directory if it doesn't exist
        base_dir = os.path.join(os.getcwd(), 'app', 'dashboards')
        user_dir = os.path.join(base_dir, str(user_id))
        os.makedirs(user_dir, exist_ok=True)
        logger.info(f"Created/verified user directory: {user_dir}")
        
        # Generate a unique filename
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"dashboard_{timestamp}.html"
        file_path = os.path.join(user_dir, filename)
        logger.info(f"Generated filepath: {file_path}")
        
        # Save the dashboard HTML
        if not save_dashboard_html(figures, file_path):
            raise Exception("Failed to save dashboard HTML")
        
        logger.info(f"Saved dashboard HTML to {file_path}")
        
        # Save to database
        dashboard_name = f"Dashboard_{timestamp}"
        try:
            dashboard_id = save_dashboard_to_db(user_id, dashboard_name, file_path)
            logger.info(f"Saved dashboard info to database with ID: {dashboard_id}")
            
            # Verify the save was successful
            with db.engine.connect() as connection:
                query = text("SELECT COUNT(*) FROM dashboard WHERE id = :dashboard_id")
                count = connection.execute(query, {"dashboard_id": dashboard_id}).scalar()
                
                if count == 0:
                    raise Exception("Dashboard was not found in database after saving")
            
            return {"success": True, "file_path": file_path, "dashboard_id": dashboard_id}
            
        except Exception as db_error:
            logger.error(f"Database error: {db_error}")
            # Clean up the file if database save failed
            if os.path.exists(file_path):
                os.remove(file_path)
            raise db_error
        
    except Exception as e:
        logger.error(f"Error saving dashboard: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return {"success": False, "error": str(e)}

def save_dashboard_to_db(user_id, dashboard_name, file_path):
    """
    Save dashboard information to the database
    """
    try:
        # Create a new dashboard entry
        new_dashboard = Dashboard(
            user_id=user_id,
            dashboard_name=dashboard_name,
            file_path=file_path
        )
        
        # Add to database
        db.session.add(new_dashboard)
        db.session.commit()
        logger.info(f"Added dashboard to database: {dashboard_name} for user {user_id}")
        
        return new_dashboard.id
        
    except Exception as e:
        logger.error(f"Error saving dashboard to database: {str(e)}")
        db.session.rollback()
        raise e

def fetch_user_dashboards(user_id):
    """
    Fetch saved dashboards for a specific user directly from the database
    """
    try:
        # Query dashboards from the database
        dashboards = Dashboard.query.filter_by(user_id=user_id).order_by(Dashboard.created_at.desc()).all()
        
        # Convert to a list of dictionaries
        result = []
        for dashboard in dashboards:
            result.append({
                "id": dashboard.id,
                "file_path": dashboard.file_path,
                "dashboard_name": dashboard.dashboard_name,
                "created_at": dashboard.created_at.isoformat() if dashboard.created_at else None
            })
        
        return result
    except Exception as e:
        logging.error(f"Error fetching dashboards from database: {e}")
        return []

def load_user_dashboard(user_id, dashboard_id):
    """
    Load a specific dashboard from the database by ID
    """
    try:
        # Query the dashboard from the database
        dashboard = Dashboard.query.get(dashboard_id)
        
        if not dashboard:
            logging.error(f"Dashboard not found: {dashboard_id}")
            return None
            
        # Check if the file exists
        if not os.path.exists(dashboard.file_path):
            logging.error(f"Dashboard file not found: {dashboard.file_path}")
            return None
            
        # Read the file contents
        with open(dashboard.file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        return content
    except Exception as e:
        logging.error(f"Error loading dashboard from file: {e}")
        return None

import requests

@app.callback(
    Output("chat-response", "children"),
    [Input("chat-button", "n_clicks")],
    [State("chat-input", "value")]
)
def ask_chatbot(n_clicks, query):
    """
    Send user query to the FastAPI chatbot endpoint and display the response.
    Enhanced with better error handling and user data isolation.
    """
    if not n_clicks or not query:
        return ""
        
    if not current_user.is_authenticated:
        return "Please log in to use the chatbot."

    # Define the FastAPI endpoint - use port 8001
    fastapi_ask_url = "http://127.0.0.1:8001/ask"
    
    try:
        # Prepare payload with user_id explicitly as string
        user_id = str(current_user.id)
        
        # Create namespace matching what fastapi_chatbot.py expects
        user_namespace = f"user_{user_id}"
        
        payload = {
            "input": query,
            "user_id": user_id,
            "session_id": user_id,
            # Include the user's namespace to ensure data isolation
            "namespace": user_namespace
        }
        
        # Log the request
        logging.info(f"Sending chat request for user {user_id} with namespace {user_namespace}")
        
        try:
            # Test connectivity before attempting the full request
            test_response = requests.get("http://127.0.0.1:8001/health", timeout=2)
            if test_response.status_code != 200:
                return dbc.Alert(
                    "Chatbot server is not responding. Please try again later.",
                    color="danger"
                )
        except:
            # If we can't connect, show a friendly message
            return dbc.Alert(
                [
                    html.H4("Chatbot Server Unavailable", className="alert-heading"),
                    html.P("The chatbot service is currently unavailable. Please make sure the FastAPI server is running on port 8001."),
                    html.Hr(),
                    html.P("To start the server, run 'uvicorn fastapi_chatbot:app --port 8001' in a separate terminal.")
                ],
                color="warning",
                dismissable=True
            )
            
        # Use requests.post with streaming and detailed error handling
        response = requests.post(
            fastapi_ask_url, 
            json=payload, 
            stream=True, 
            timeout=60
        )
        
        # Check for successful response
        if response.status_code != 200:
            # Try to extract more detailed error information
            try:
                error_data = response.json()
                error_details = error_data.get('detail', response.text)
            except:
                error_details = response.text or f"Error code: {response.status_code}"
            
            # Check for Pinecone-specific errors
            if "no data found for your account" in error_details.lower() or "no vectors" in error_details.lower():
                return dbc.Alert(
                    [
                        html.H4("No Data Found", className="alert-heading"),
                        html.P("You need to upload and ingest a CSV file before asking questions."),
                        html.Hr(),
                        html.P("Use the 'Ingest CSV to Pinecone' button after uploading your file.")
                    ],
                    color="info",
                    dismissable=True
                )
            
            return dbc.Alert(
                [
                    html.H4("Chat Request Failed", className="alert-heading"),
                    html.P(f"Error: {error_details}"),
                    html.Hr(),
                    html.P("Please try a different question or check if you've uploaded data.")
                ],
                color="danger",
                dismissable=True
            )

        # Collect and process streaming response
        full_text = ""
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                try:
                    decoded_chunk = chunk.decode("utf-8", errors="ignore")
                    full_text += decoded_chunk
                except Exception as decode_err:
                    logging.warning(f"Error decoding chunk: {decode_err}")
                    # Continue processing other chunks
        
        # Check if the response was empty or the generic error message
        if not full_text or "unable to provide" in full_text.lower():
            return dbc.Alert(
                [
                    html.H4("Unable to Answer", className="alert-heading"),
                    html.P("The AI couldn't retrieve specific data from your dataset to answer this question."),
                    html.Hr(),
                    html.P([
                        "Try asking more specific questions about your data. For example:",
                        html.Ul([
                            html.Li("What are my top selling products?"),
                            html.Li("Which department has the highest revenue?"),
                            html.Li("Show me sales trends by month"),
                        ])
                    ])
                ],
                color="warning",
                dismissable=True
            )

        # Format the response for display
        formatted_response = html.Div([
            html.H4("AI Response:", style={"color": "#8ac4ff", "marginBottom": "10px"}),
            html.Div(
                dcc.Markdown(full_text, 
                    style={
                        "backgroundColor": "#1d1b31", 
                        "padding": "15px", 
                        "borderRadius": "5px", 
                        "whiteSpace": "pre-wrap"
                    }
                )
            )
        ])

        # Save the chat history to the database
        try:
            new_chat = ChatHistory(
                user_id=current_user.id, 
                query=query, 
                response=full_text
            )
            db.session.add(new_chat)
            db.session.commit()
            logging.info(f"Chat history saved for user {user_id}")
        except Exception as db_error:
            logging.error(f"Error saving chat history: {db_error}")
            # Non-critical error, continue to show response

        return formatted_response

    except requests.exceptions.Timeout:
        logging.error(f"Timeout for chat request from user {user_id}")
        return dbc.Alert(
            "The request timed out. Your question may be too complex or the server is busy.",
            color="warning",
            dismissable=True
        )
        
    except requests.exceptions.ConnectionError:
        logging.error(f"Connection error for chat request from user {user_id}")
        return dbc.Alert(
            "Could not connect to the chatbot server. Please try again later.",
            color="danger",
            dismissable=True
        )

    except requests.exceptions.RequestException as req_error:
        # Handle specific request-related exceptions
        error_message = f"Request failed: {str(req_error)}"
        logging.error(f"Request error for user {user_id}: {error_message}")
        return dbc.Alert(
            error_message,
            color="danger",
            dismissable=True
        )
        
    except Exception as e:
        # Catch all other exceptions
        logging.error(f"Unexpected error in ask_chatbot: {e}")
        return dbc.Alert(
            f"An unexpected error occurred: {str(e)}",
            color="danger",
            dismissable=True
        )

# Add these new callbacks to display saved dashboards and chat history
@app.callback(
    Output("saved-dashboards", "children"),
    [Input("refresh-history-button", "n_clicks"),
     Input('interval-component', 'n_intervals')],
    prevent_initial_call=False  # Allow initial call to populate the dashboards list
)
def display_saved_dashboards(n_clicks, n_intervals):
    """
    Display the user's saved dashboards as clickable list items.
    """
    # Get the triggered component
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Skip automatic interval refreshes when there are no new dashboards
    if triggered_id == 'interval-component' and n_intervals > 1:
        # Only refresh if there's a chance of new data
        # Get timestamp from 30 seconds ago
        recent_time = datetime.now() - timedelta(seconds=60)
        
        try:
            with db.engine.connect() as connection:
                query = text("""
                    SELECT COUNT(*) FROM dashboard 
                    WHERE user_id = :user_id AND created_at > :recent_time
                """)
                recent_count = connection.execute(
                    query, 
                    {"user_id": current_user.id, "recent_time": recent_time}
                ).scalar()
                
                # If no new dashboards in the last minute, skip the refresh
                if recent_count == 0:
                    # Return no_update to prevent a refresh
                    raise dash.exceptions.PreventUpdate
        except:
            # If any error occurs during this check, continue with the refresh
            pass
    
    try:
        if not current_user.is_authenticated:
            return html.Li("Please log in to view saved dashboards", style={"color": "#ff9999"})

        # Try direct database query with error handling
        try:
            # Use raw SQL to prevent errors with ORM model mismatch
            with db.engine.connect() as connection:
                query = text("SELECT id, file_path, created_at FROM dashboard WHERE user_id = :user_id")
                result = connection.execute(query, {"user_id": current_user.id})
                dashboards = [{"id": row[0], "file_path": row[1], "created_at": row[2]} for row in result]
        except Exception as sql_error:
            logging.error(f"Error querying dashboard table: {sql_error}")
            return html.Li(f"Database error: {str(sql_error)}", style={"color": "#ff9999"})
        
        if not dashboards:
            return html.Li("No saved dashboards found", style={"color": "#aaaaaa"})
        
        dashboard_items = []
        
        for dashboard in dashboards:
            # Create path for viewing the dashboard
            file_path = dashboard["file_path"]
            # Ensure the file exists
            if os.path.exists(file_path):
                # Create URL that will serve the dashboard HTML
                dashboard_url = f"/view_dashboard/{dashboard['id']}"
                # Use a default name if dashboard_name isn't available
                created_at = dashboard.get("created_at", datetime.now())
                if isinstance(created_at, str):
                    try:
                        created_at = datetime.fromisoformat(created_at)
                    except:
                        created_at = datetime.now()
                
                dashboard_name = f"Dashboard {created_at.strftime('%Y-%m-%d %H:%M')}"
                
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
                        f"Created: {created_at.strftime('%Y-%m-%d %H:%M')}",
                        style={"color": "#aaaaaa", "marginLeft": "15px"}
                    )
                ], style={"marginBottom": "10px"})
                
                dashboard_items.append(item)
        
        if not dashboard_items:
            return html.Li("No readable dashboard files found", style={"color": "#aaaaaa"})
            
        return dashboard_items
        
    except Exception as e:
        logging.error(f"Error displaying saved dashboards: {e}")
        return html.Li(f"Error loading dashboards: {str(e)}", style={"color": "#ff9999"})

@app.callback(
    Output("past-chats", "children"),
    [Input("refresh-history-button", "n_clicks"),
     Input("interval-component", "n_intervals")],
    prevent_initial_call=False  # Allow initial call to populate the chat history
)
def update_chat_history(n_clicks=None, n_intervals=None):
    try:
        if not current_user.is_authenticated:
            return [html.Div("Please log in to view chat history", style={"color": "#aaa"})]
        
        user_id = str(current_user.id)
        
        # Query the chat_history document for this user
        try:
            doc = mongo_db["chat_history"].find_one({"user_id": user_id})
            if not doc:
                return [html.Div("No chat history", style={"color": "#aaa"})]
            
            full_chat = doc.get("chat_history", [])
            if not full_chat:
                return [html.Div("No chat history", style={"color": "#aaa"})]
            
            chat_items = []
            for entry in full_chat:
                role = entry.get("role", "")
                content = entry.get("content", "")
                
                # Use improved chat bubble styling
                if role == "user":
                    bubble_style = {
                        "backgroundColor": "#f1f0f0",
                        "color": "#000",
                        "padding": "12px 16px",
                        "borderRadius": "15px",
                        "maxWidth": "70%",
                        "marginBottom": "8px",
                        "alignSelf": "flex-end",
                        "boxShadow": "2px 2px 5px rgba(0,0,0,0.15)"
                    }
                else:
                    bubble_style = {
                        "backgroundColor": "#f5f507",
                        "color": "#000",
                        "padding": "12px 16px",
                        "borderRadius": "15px",
                        "maxWidth": "70%",
                        "marginBottom": "8px",
                        "alignSelf": "flex-start",
                        "boxShadow": "2px 2px 5px rgba(0,0,0,0.15)"
                    }
                
                chat_block = html.Div([
                    html.Div(content, style=bubble_style)
                ], style={
                    "display": "flex",
                    "flexDirection": "column",
                    "marginBottom": "15px",
                    "padding": "10px"
                })
                chat_items.append(chat_block)
            
            return chat_items
            
        except Exception as mongo_error:
            print(f"MongoDB error: {mongo_error}")
            return [html.Div("Error accessing chat history", style={"color": "red"})]
    
    except Exception as e:
        print(f"Error updating chat history: {e}")
        return [html.Div(f"Error fetching chat history: {str(e)}", style={"color": "red"})]

# # Remove duplicate route and keep only one implementation
# @app.server.route('/dash/load_dashboard/<dashboard_name>')
# def serve_dashboard(dashboard_name):
#     if not current_user.is_authenticated:
#         return redirect(url_for('login'))
    
#     try:
#         # Load the dashboard content
#         dashboard_content = load_user_dashboard(str(current_user.id), dashboard_name)
        
#         if dashboard_content:
#             return dashboard_content
#         else:
#             return "Dashboard not found or could not be loaded", 404
#     except Exception as e:
#         print(f"Error serving dashboard: {e}")
#         return f"Error loading dashboard: {str(e)}", 500

# # Route to handle loading saved dashboards
# @app.server.route('/dash/load_dashboard/<dashboard_name>')
# def serve_dashboard(dashboard_name):
#     if not current_user.is_authenticated:
#         return redirect(url_for('login'))
    
#     # Load the dashboard content
#     dashboard_content = load_user_dashboard(str(current_user.id), dashboard_name)
    
#     if dashboard_content:
#         return dashboard_content
#     else:
#         return "Dashboard not found or could not be loaded", 404

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
@cache.memoize(timeout=1000)
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
    print("DEBUG: display_filename callback triggered")
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
    Fixed to properly handle user namespaces.
    """
    if not n_clicks:
        return ""
        
    if not contents:
        return "No CSV uploaded yet. Please upload a CSV file first."
        
    if not current_user.is_authenticated:
        return "Please log in to ingest data."
    
    try:
        # Set up the API endpoint - ensure port matches FastAPI server
        fastapi_url = "http://127.0.0.1:8001/ask"
        fastapi_url = "http://127.0.0.1:8001/upload_csv"  # Make sure URL ends with upload_csv
        
        # Get user identification - ALWAYS use string format
        user_id = str(current_user.id)
        
        # Extract base64 data from the contents
        try:
            b64_data = contents.split(",")[1]
        except (IndexError, AttributeError) as e:
            logging.error(f"Error extracting base64 data: {e}")
            return "Error processing CSV data. Please try a different file."
        
        # Build the payload with user_id - make sure it's a string
        payload = {
            "user_id": user_id,
            "session_id": user_id,
            "csv_base64": b64_data
        }
        
        # Log the request
        logging.info(f"Sending CSV ingestion request for user {user_id}")
        
        # Make the request with proper error handling
        try:
            response = requests.post(
                fastapi_url, 
                json=payload, 
                timeout=60  # Increased timeout for larger files
            )
            
            # Handle successful response
            if response.status_code == 200:
                try:
                    resp_data = response.json()
                    rows_ingested = resp_data.get('rows_ingested', 'unknown')
                    namespace = resp_data.get('namespace', f"user_{user_id}")
                    
                    # Create a success message with helpful information
                    success_message = dbc.Alert(
                        [
                            html.H4("CSV Ingestion Successful", className="alert-heading"),
                            html.P(f"Rows ingested: {rows_ingested}"),
                            html.P(f"Your data is stored in namespace: {namespace}"),
                            html.P(f"Your data is ready for querying in the chat section below.")
                        ],
                        color="success",
                        dismissable=True
                    )
                    
                    logging.info(f"Successful ingestion for user {user_id}: {rows_ingested} rows in namespace {namespace}")
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
     Output('graphs-container', 'children'),
     Output('save-status', 'children')],  # Add output for save status
    [Input('submit-button', 'n_clicks'),
     Input('save-dashboard-button', 'n_clicks')],  # Add save button input
    [State('upload-data', 'contents'),
     State('data-aspect', 'value'),
     State('range-interest', 'value'),
     State('story-goal', 'value'),
     State('param-check', 'value'),
     State('viz-priority', 'value'),
     State('sample-size', 'value'),
     State('color-scheme', 'value'),
     State('graphs-container', 'children')]  # Add this to capture existing figures
)
def update_graph(n_clicks, save_clicks, contents, aspect, range_interest, story_goal, param_check, viz_priority, sample_size, color_scheme, existing_graphs):
    """
    Generate visualizations based on user inputs and save dashboard when requested.
    """
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else 'No clicks'
    
    # Initialize outputs
    summary_output = html.Div("No data analyzed yet")
    graphs_output = html.Div("No visualizations generated")
    save_status = ""
    
    # Check if we have an authenticated user
    if not current_user.is_authenticated:
        return summary_output, graphs_output, html.Div("Please login to analyze data", style={"color": "red"})
    
    # Save dashboard button was clicked
    if triggered_id == 'save-dashboard-button' and save_clicks:
        # Get the currently displayed figures
        figures = []
        
        # Check if we have existing graphs and extract figures
        if existing_graphs and not isinstance(existing_graphs, str):
            logging.info(f"Found existing graphs: {type(existing_graphs)}")
            
            # If it's a single row
            if isinstance(existing_graphs, dict) and 'props' in existing_graphs:
                row_children = existing_graphs.get('props', {}).get('children', [])
                if isinstance(row_children, list):
                    for col in row_children:
                        if isinstance(col, dict) and 'props' in col:
                            col_children = col.get('props', {}).get('children')
                            if isinstance(col_children, dict) and 'props' in col_children:
                                fig_props = col_children.get('props', {})
                                if 'figure' in fig_props:
                                    figures.append(fig_props['figure'])
            
            # If it's a list of rows
            elif isinstance(existing_graphs, list):
                for row in existing_graphs:
                    if isinstance(row, dict) and 'props' in row:
                        row_children = row.get('props', {}).get('children', [])
                        if isinstance(row_children, list):
                            for col in row_children:
                                if isinstance(col, dict) and 'props' in col:
                                    col_children = col.get('props', {}).get('children')
                                    if isinstance(col_children, dict) and 'props' in col_children:
                                        fig_props = col_children.get('props', {})
                                        if 'figure' in fig_props:
                                            figures.append(fig_props['figure'])
        
        logging.info(f"Extracted {len(figures)} figures for saving")
        
        if not figures:
            # Try a different approach - create fresh figures
            try:
                if contents:
                    df = parse_contents(contents)
                    if df is not None:
                        # Get a sample of the data
                        df = get_sample(df, sample_size or 1000)
                        
                        # Create basic visualizations
                        figures = []
                        
                        # Add a simple bar chart of the sum by category if possible
                        if len(df.select_dtypes(include=['object']).columns) > 0 and len(df.select_dtypes(include=['number']).columns) > 0:
                            cat_col = df.select_dtypes(include=['object']).columns[0]
                            num_col = df.select_dtypes(include=['number']).columns[0]
                            
                            fig = px.bar(
                                df.groupby(cat_col)[num_col].sum().reset_index(),
                                x=cat_col,
                                y=num_col,
                                title=f'Sum of {num_col} by {cat_col}'
                            )
                            figures.append(fig)
                        
                        # Add a simple line chart if date column exists
                        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                        if date_cols and len(df.select_dtypes(include=['number']).columns) > 0:
                            date_col = date_cols[0]
                            num_col = df.select_dtypes(include=['number']).columns[0]
                            
                            # Convert to datetime if not already
                            try:
                                df[date_col] = pd.to_datetime(df[date_col])
                                fig = px.line(
                                    df.groupby(date_col)[num_col].sum().reset_index(),
                                    x=date_col,
                                    y=num_col,
                                    title=f'Trend of {num_col} over time'
                                )
                                figures.append(fig)
                            except:
                                pass
                    
                    logging.info(f"Created {len(figures)} fallback figures for saving")
            except Exception as e:
                logging.error(f"Error creating fallback figures: {e}")
        
        if not figures:
            return summary_output, graphs_output, html.Div("No visualizations to save. Please generate visualizations first.", style={"color": "orange"})
        
        try:
            # Save dashboard
            result = save_dashboard_for_user(current_user.id, figures)
            
            if result.get("success", False):
                # Verify the dashboard was saved by checking the database
                with db.engine.connect() as connection:
                    query = text("SELECT COUNT(*) FROM dashboard WHERE user_id = :user_id")
                    count = connection.execute(query, {"user_id": current_user.id}).scalar()
                
                if count > 0:
                    save_status = html.Div("Dashboard saved successfully! View it in your Saved Dashboards.", 
                                         style={"color": "green", "margin": "10px 0"})
                else:
                    save_status = html.Div("Dashboard was processed but not found in database. Please try again.", 
                                         style={"color": "orange", "margin": "10px 0"})
            else:
                save_status = html.Div(f"Error saving dashboard: {result.get('error', 'Unknown error')}", 
                                     style={"color": "red", "margin": "10px 0"})
                
        except Exception as e:
            logging.error(f"Error saving dashboard: {e}")
            save_status = html.Div(f"Error saving dashboard: {str(e)}", 
                                 style={"color": "red", "margin": "10px 0"})
        
        return summary_output, existing_graphs, save_status
    
    # Submit button clicked or other triggers
    if not contents or not n_clicks:
        return summary_output, graphs_output, save_status
    
    # Main visualization creation code
    try:
        df = parse_contents(contents)
        if df is not None:
            # Possibly handle_missing_values(df) if you want
            df_original = df.copy()
            df = get_sample(df, sample_size or 1000)  # Default to 1000 if not specified

            # 1) Summaries
            summary_cards = generate_statistical_cards(df_original)

            # 2) Use GPT to suggest cleaning steps & visuals
            prompt = generate_prompt(df, aspect, range_interest, story_goal, param_check, viz_priority)
            cleaning_steps, visualization_suggestions, ml_steps = query_openai(prompt)
            
            df = apply_data_cleaning(df, cleaning_steps)
            if df.empty:
                logging.warning("DataFrame empty after cleaning. No visuals to show.")
                return summary_cards, html.Div("No data available after cleaning.", style={'color': 'red'}), save_status

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

            return summary_cards, rows, save_status
        else:
            logging.error("Could not parse contents of CSV")
            return html.Div("Error: Could not parse the uploaded file."), html.Div(), save_status
    except Exception as e:
        logging.error(f"Error in update_graph: {e}")
        return html.Div(f"Error: {str(e)}"), html.Div(), save_status

# Modify save_dashboard_html function to be more flexible
def save_dashboard_html(figures, file_path):
    """Save a list of plotly figures as an HTML dashboard using proper Plotly figure saving."""
    try:
        import plotly.graph_objects as go
        from jinja2 import Template
        
        # Create a template for the dashboard
        template = """
        <!DOCTYPE html>
        <html>
            <head>
                <title>Dashboard</title>
                <script src='https://cdn.plot.ly/plotly-latest.min.js'></script>
                <style>
                    body { margin: 0; padding: 20px; font-family: Arial, sans-serif; background-color: #081028; color: white; }
                    .dashboard { display: flex; flex-wrap: wrap; justify-content: center; }
                    .graph-container { width: 90%; max-width: 800px; margin: 15px auto; padding: 10px; background-color: #1d1b31; border-radius: 8px; }
                    h1 { text-align: center; color: #fff; }
                </style>
            </head>
            <body>
                <h1>Data Visualization Dashboard</h1>
                <div class="dashboard">
                    {% for fig in figures %}
                    <div class="graph-container">
                        {{ fig }}
                    </div>
                    {% endfor %}
                </div>
            </body>
        </html>
        """
        
        # Convert figure dictionaries to Plotly figures and generate HTML
        figure_htmls = []
        for fig_dict in figures:
            if isinstance(fig_dict, dict):
                # Create a Plotly figure from the dictionary
                fig = go.Figure(fig_dict)
                # Generate HTML for the figure
                figure_htmls.append(fig.to_html(full_html=False, include_plotlyjs=False))
            else:
                # If it's already a Plotly figure
                figure_htmls.append(fig_dict.to_html(full_html=False, include_plotlyjs=False))
        
        # Render the template with the figures
        j2_template = Template(template)
        html_content = j2_template.render(figures=figure_htmls)
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return True
    except Exception as e:
        logging.error(f"Error saving dashboard: {str(e)}")
        return False

# Add a clientside callback to help with user authentication
app.clientside_callback(
    """
    function(n_intervals) {
        console.log("Clientside callback triggered");
        return document.querySelector('body').getAttribute('data-user-id') || "0";
    }
    """,
    Output('current-user-store', 'children'),
    Input('url', 'pathname')
)

# Also add data-user-id to the body tag
@app.server.after_request
def add_user_id_to_body(response):
    if hasattr(current_user, 'is_authenticated') and current_user.is_authenticated:
        print(f"DEBUG: Adding user_id={current_user.id} to body tag")
        try:
            if response.content_type and 'text/html' in response.content_type:
                response_data = response.get_data(as_text=True)
                # Add data-user-id attribute to the body tag
                modified_data = response_data.replace('<body', f'<body data-user-id="{current_user.id}"')
                response.set_data(modified_data)
        except Exception as e:
            print(f"DEBUG: Error modifying response: {str(e)}")
    return response

# ------------------------------------------------
# 10) Main
# ------------------------------------------------
if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info("Registered routes:")
    for rule in server.url_map.iter_rules():
        logger.info(f"Route: {rule}")
    server.run(debug=True, host='localhost', port=5001, use_reloader=False)

@app.server.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

def create_html_download_button(figs, file_name="plotly_graph", button_name="Download as HTML"):
    """
    Given a Plotly Figure or list of Plotly Figures, creates a download button that saves the figure(s) as an HTML file.
    
    :param figs: Plotly Figure or List of Plotly Figures
    :param file_name: String (name of file to download >> '.html' will be appended)
    :param button_name: String (text to display on the button)
    :return: Dash dbc.Button object
    """
    import io
    from base64 import b64encode
    
    # Special handling of storing multiple figures in buffer
    if isinstance(figs, list) and len(figs) > 1:
        # Keep non-None figures
        figs = [fig for fig in figs if fig is not None]
        # Create buffer
        main_buffer = io.StringIO()
        outputs = []
        # Write first figure with full HTML
        _buffer = io.StringIO()
        figs[0].write_html(_buffer, full_html=True, include_plotlyjs='cdn')
        outputs.append(_buffer)
        # Write remaining figures as divs
        for fig in figs[1:]:
            _buffer = io.StringIO()
            fig.write_html(_buffer, full_html=False)
            outputs.append(_buffer)
        
        # Concatenate all outputs
        main_buffer.write(''.join([i.getvalue() for i in outputs]))
    else:
        # Create buffer for single figure
        main_buffer = io.StringIO()
        # Write figure to buffer
        if isinstance(figs, list):
            figs[0].write_html(main_buffer)
        else:
            figs.write_html(main_buffer)
    
    # Convert buffer to bytes and encode
    html_bytes = main_buffer.getvalue().encode()
    encoded = b64encode(html_bytes).decode()
    
    # Create download button
    download_html = dbc.Button(
        button_name,
        href="data:text/html;base64," + encoded,
        download=file_name + ".html",
        color="primary",
        className="mt-2"
    )
    
    return download_html

# Add this function near the top of the file with other utility functions
def check_dashboard_table():
    """Check and display contents of the dashboard table"""
    try:
        with db.engine.connect() as connection:
            result = connection.execute(text("SELECT id, user_id, file_path, dashboard_name FROM dashboard"))
            rows = result.fetchall()
            
            if not rows:
                print("No dashboards found in the database")
                return
                
            print("\nDashboard Table Contents:")
            print("-" * 80)
            print(f"{'ID':<5} {'User ID':<8} {'Dashboard Name':<30} {'File Path'}")
            print("-" * 80)
            
            for row in rows:
                print(f"{row[0]:<5} {row[1]:<8} {row[3][:30]:<30} {row[2]}")
            print("-" * 80)
            
    except Exception as e:
        print(f"Error checking dashboard table: {e}")