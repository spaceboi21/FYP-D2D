import os
import sys

# Add app directory to Python path
app_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app'))
sys.path.append(app_dir)

# Monkey patch the module system
import builtins
original_import = builtins.__import__

def patched_import(name, *args, **kwargs):
    if name == 'fastapi_chatbot':
        return original_import('app.fastapi_chatbot', *args, **kwargs)
    return original_import(name, *args, **kwargs)

builtins.__import__ = patched_import

# Now import from new_ver with the Flask app
from app.new_ver import db, User, server as flask_app

# Create all tables within the application context
print("Creating database tables...")
with flask_app.app_context():
    db.create_all()
    print("Database tables created.")

    # List all tables
    try:
        print("Tables in database:", db.engine.table_names())
    except Exception as e:
        print(f"Could not list tables: {e}") 