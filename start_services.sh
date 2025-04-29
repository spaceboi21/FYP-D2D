#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if screen is installed
if ! command_exists screen; then
    echo "Installing screen..."
    sudo apt-get update
    sudo apt-get install -y screen
fi

# Create a new screen session for each service
echo "Starting services..."

# Start Flask app
screen -dmS flask_app bash -c 'cd "$(dirname "$0")" && python app/new_ver.py'

# Start FastAPI app
screen -dmS fastapi_app bash -c 'cd "$(dirname "$0")" && python -m uvicorn app.fastapi_chatbot:app --host 0.0.0.0 --port 8001 --reload'

# Start React app
screen -dmS react_app bash -c 'cd "$(dirname "$0")/my-react-app" && npm start'

echo "All services started in screen sessions."
echo "To view a session, use: screen -r [session_name]"
echo "Available sessions:"
echo "- flask_app"
echo "- fastapi_app"
echo "- react_app"
echo ""
echo "To detach from a screen session, press Ctrl+A followed by D"
echo "To list all screen sessions, use: screen -ls"
echo "To kill all screen sessions, use: screen -X quit" 