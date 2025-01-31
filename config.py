class Config:
    # 智谱AI配置（带权重示例）
    ZHIPU_CONFIG = {
        'api_keys': [
            {'key': 'your apiKey', 'weight': 1.0}
        ],
        'model_weight': 1,
        'max_concurrency': 200
    }

    # 系统配置
    DEBUG = True
    FLASK_ENV = 'development'
    ENABLED_MODELS = ['zhipu']