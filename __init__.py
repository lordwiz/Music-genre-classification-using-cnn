# __init__.py
from flask import Flask

def create_app():
    app = Flask(__name__)

    # Register blueprints and configure app if needed

    return app

# Import routes to ensure they are registered with the app
from flask import routes
