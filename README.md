# AI Data Visualization Dashboard

A Flask and Dash application that provides data visualization and analysis capabilities with AI-powered insights.

## Features

- User authentication (login/signup)
- CSV file upload and analysis
- AI-powered data visualization recommendations
- Interactive dashboards
- Chat interface for data queries
- Dashboard saving and loading

## Tech Stack

- **Backend**: Flask, FastAPI
- **Frontend**: Dash, HTML, CSS
- **Database**: MongoDB, SQLite
- **AI/ML**: OpenAI GPT, Pinecone Vector Database
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Plotly

## Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file
4. Run the application: `python app/new_ver.py`

## Project Structure

- `app/fastapi_chatbot.py`: FastAPI backend for chatbot functionality
- `app/new_ver.py`: Main Flask and Dash application
- `app/setup_db.py`: Database setup and initialization
- `app/templates/`: HTML templates
- `app/static/`: Static assets (CSS, JS, images)
- `app/instance/`: Instance-specific files
- `app/saved_dashboards/`: Saved user dashboards
- `app/user_chats/`: User chat history

## License

MIT 