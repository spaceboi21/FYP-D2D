from testing import db, server

with server.app_context():
    db.create_all()
    print("✅ Database has been initialized!")