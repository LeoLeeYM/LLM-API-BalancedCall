from .load_balancer import LoadBalancer
from .models import MODEL_CLASSES

class LLMManager:
    def __init__(self, config):
        self.models = self._init_models(config)
        self.load_balancer = LoadBalancer()

    def _init_models(self, config):
        models = {}
        for model_name in config['ENABLED_MODELS']:
            if model_name not in MODEL_CLASSES:
                continue
            model_class = MODEL_CLASSES[model_name]
            models[model_name] = model_class(config)
        return models

    def process_request(self, messages, tools=None):
        model, key = self._select_instance(tools)
        return model.chat_completion(messages, tools, key)

    def process_stream_request(self, messages, tools=None):
        model, key = self._select_instance(tools)
        return model.chat_completion(messages, tools, key, stream=True)

    def _select_instance(self, requires_tools):
        return self.load_balancer.select_instance(
            self.models, 
            requires_tools=requires_tools
        )

    def get_system_capacity(self):
        capacity = {
            'total': {'concurrency': 0, 'qps': 0},
            'models': {}
        }
        
        for name, model in self.models.items():
            info = model.get_capacity_info()
            if not info:
                continue
                
            current, max_cap = info
            model_data = {
                'type': model.strategy.capacity_type,
                'current': current,
                'max': max_cap,
                'weight': model.model_weight,
                'keys': [
                    {
                        'key': k['key'],
                        'weight': k['weight'],
                        'current': model.get_key_load(k['key'])
                    }
                    for k in model.api_keys
                ]
            }
            capacity['models'][name] = model_data
            capacity['total'][model_data['type']] += max_cap
            
        return capacity

    def get_key_load(self, model_name, api_key):
        return self.models[model_name].get_key_load(api_key)
    
    def get_system_load(self):
        total_load = 0
        total_capacity = 0
        
        for model in self.models.values():
            current_load, max_capacity = model.get_load_status()
            total_load += current_load
            total_capacity += max_capacity
        
        return (total_load / total_capacity * 100) if total_capacity > 0 else 0
    
    def get_model_capacity_info(self, model_name):
        """获取指定模型的容量信息"""
        model = self.models.get(model_name)
        if not model:
            return None
        return model.get_capacity_info()