from flask_cors import CORS

def configure_cors(app):
    CORS(app, resources={
        r"/*": {
            "origins": [
                "http://localhost:3000",  # React frontend
                "http://localhost:5000",  # Main backend
                "http://localhost:4242",  # Stripe server
                "http://localhost:5050"   # Additional backend port
            ],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": [
                "Content-Type",
                "Authorization",
                "Access-Control-Allow-Origin",
                "Access-Control-Allow-Headers",
                "Access-Control-Allow-Methods"
            ],
            "supports_credentials": True
        }
    })
    return app 