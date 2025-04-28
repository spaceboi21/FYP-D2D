#!/bin/bash

# Kill any existing processes on the required ports
echo "Cleaning up existing processes..."
pkill -f "ngrok"
pkill -f "node"
pkill -f "python.*new_ver.py"
pkill -f "python.*stripe_server.py"

# Set environment variables for CORS
export FLASK_ENV=development
export FLASK_DEBUG=1

# Start the React frontend
echo "Starting React frontend..."
cd my-react-app/my-react-app
npm start &
REACT_PID=$!
cd ../..

# Start the FastAPI backend
echo "Starting FastAPI backend..."
cd app
python new_ver.py &
BACKEND_PID=$!
cd ..

# Start the Stripe server
echo "Starting Stripe server..."
cd app
python stripe_server.py &
STRIPE_PID=$!
cd ..

# Start ngrok for the backend
echo "Starting ngrok..."
ngrok http 5000 &
NGROK_PID=$!

# Wait for ngrok to start and get the URL
sleep 5
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | jq -r '.tunnels[0].public_url')

echo "============================================="
echo "Demo is running!"
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:5000"
echo "Ngrok URL: $NGROK_URL"
echo "Stripe Server: http://localhost:4242"
echo "============================================="
echo "Press Ctrl+C to stop all services"

# Keep the script running
trap "kill $REACT_PID $BACKEND_PID $STRIPE_PID $NGROK_PID; exit" INT
wait 