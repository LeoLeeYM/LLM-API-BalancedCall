from flask import Flask
from config import Config

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    
    # 初始化组件
    _register_blueprints(app)
    _init_managers(app)
    
    return app

def _register_blueprints(app):
    from app.llm.routes import llm_blueprint
    app.register_blueprint(llm_blueprint, url_prefix='/llm')

def _init_managers(app):
    from app.utils.llm_manager import LLMManager
    app.llm_manager = LLMManager(app.config)