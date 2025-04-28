# Import all models (User, Dashboard, ChatHistory, etc.)
from new_ver import server, db
 # This ensures all models in dash_flask_app.py are registered with SQLAlchemy.

import os
from datetime import datetime

# Create multiple folders for user-specific storage
DASHBOARD_FOLDER = "saved_dashboards"
USER_CHAT_FOLDER = "user_chats"

def initialize_storage_structure():
    """
    Create necessary folders for user-specific storage
    """
    folders_to_create = [
        DASHBOARD_FOLDER,
        USER_CHAT_FOLDER
    ]
    
    for folder in folders_to_create:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Created {folder} directory")


class ChatHistory(db.Model):
    __tablename__ = 'chat_history'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    query = db.Column(db.Text, nullable=False)
    response = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Define relationship with User model if needed
    user = db.relationship('User', backref=db.backref('chat_history', lazy=True))
    
    def __repr__(self):
        return f"<ChatHistory {self.id} - {self.query[:20]}...>"

with server.app_context():
    # Create database tables
    db.create_all()
    print("✅ Database tables initialized!")
    
    # Initialize storage structure
    initialize_storage_structure()
    print("✅ User storage directories prepared!")