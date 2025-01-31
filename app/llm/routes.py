from flask import Blueprint, request, jsonify, Response, current_app
from datetime import datetime

llm_blueprint = Blueprint('llm', __name__)

@llm_blueprint.route('/chat', methods=['POST'])
def handle_chat():
    try:
        data = request.get_json()
        messages = data['messages']
        tools = data.get('tools')
        
        result = current_app.llm_manager.process_request(messages, tools)
        return jsonify({"result": result}), 200
    except KeyError:
        return jsonify({"error": "Missing required field 'messages'"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@llm_blueprint.route('/chat/stream', methods=['POST'])
def handle_chat_stream():
    try:
        data = request.get_json()
        messages = data['messages']
        tools = data.get('tools')
        
        generator = current_app.llm_manager.process_stream_request(messages, tools)
        return Response(generator, mimetype='text/event-stream')
    except KeyError:
        return jsonify({"error": "Missing required field 'messages'"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@llm_blueprint.route('/system-load', methods=['GET'])
def get_system_load():
    load = current_app.llm_manager.get_system_load()
    return jsonify({"load_percent": round(load, 2)}), 200

@llm_blueprint.route('/system-capacity', methods=['GET'])
def get_system_capacity():
    try:
        capacity_info = current_app.llm_manager.get_system_capacity()
        return jsonify(capacity_info), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@llm_blueprint.route('/model-load/<model_name>', methods=['GET'])
def get_model_load(model_name):
    """获取指定模型的负载详情"""
    try:
        manager = current_app.llm_manager
        if model_name not in manager.models:
            return jsonify({"error": f"Model {model_name} not found"}), 404
        
        model = manager.models[model_name]
        capacity_info = model.get_capacity_info()
        
        return jsonify({
            "model": model_name,
            "type": capacity_info['type'],
            "current": capacity_info['current'],
            "max": capacity_info['max'],
            "weight": capacity_info['weight'],
            "keys": capacity_info['keys']
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@llm_blueprint.route('/key-load/<model_name>/<api_key>', methods=['GET'])
def get_key_load(model_name, api_key):
    """获取指定密钥的负载详情"""
    try:
        manager = current_app.llm_manager
        if model_name not in manager.models:
            return jsonify({"error": f"Model {model_name} not found"}), 404
        
        model = manager.models[model_name]
        key_info = next((k for k in model.api_keys if k['key'] == api_key), None)
        
        if not key_info:
            return jsonify({"error": f"API key {api_key} not found"}), 404
        
        current_load = model.get_key_load(api_key)
        _, max_cap = model.get_load_status()
        
        return jsonify({
            "model": model_name,
            "key": api_key,
            "weight": key_info['weight'],
            "current": current_load,
            "max": max_cap // len(model.api_keys)  # 单key最大容量
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@llm_blueprint.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }), 200

# @llm_blueprint.route('/models', methods=['GET'])
# def get_enabled_models():
#     """获取已启用模型列表"""
#     try:
#         enabled_models = current_app.config['ENABLED_MODELS']
#         return jsonify({"models": enabled_models}), 200
#     except KeyError:
#         return jsonify({"error": "Configuration error"}), 500

# @llm_blueprint.route('/keys/<model_name>', methods=['GET'])
# def get_model_keys(model_name):
#     """获取指定模型的密钥列表"""
#     try:
#         config_section = f"{model_name.upper()}_CONFIG"
#         model_config = current_app.config.get(config_section)
        
#         if not model_config:
#             return jsonify({"error": f"Config for {model_name} not found"}), 404
            
#         return jsonify({
#             "model": model_name,
#             "keys": [
#                 {"key": k['key'], "weight": k.get('weight', 1.0)} 
#                 for k in model_config['api_keys']
#             ]
#         }), 200
#     except KeyError as e:
#         return jsonify({"error": f"Invalid config format: {str(e)}"}), 500