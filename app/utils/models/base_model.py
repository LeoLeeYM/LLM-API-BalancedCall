from abc import ABC, abstractmethod
from app.utils.load_strategies import ConcurrencyStrategy, QPSStrategy

class BaseModel(ABC):
    STRATEGY_CLASS = None
    CONFIG_SECTION = None
    supports_tools = False

    def __init__(self, config):
        if not self.CONFIG_SECTION:
            raise ValueError("CONFIG_SECTION must be defined")
            
        self.config = config[self.CONFIG_SECTION]
        self.api_keys = self._parse_api_keys()
        self.model_weight = self.config.get('model_weight', 1.0)
        self.strategy = self.STRATEGY_CLASS(
            api_keys=self.api_keys,
            **self._get_strategy_params()
        )

    def _parse_api_keys(self):
        keys = []
        for item in self.config['api_keys']:
            if isinstance(item, dict):
                keys.append({'key': item['key'], 'weight': item.get('weight', 1.0)})
            else:
                keys.append({'key': item, 'weight': 1.0})
        return keys

    def _get_strategy_params(self):
        raise NotImplementedError

    # 新增关键方法：将策略类方法暴露给模型实例
    def get_available_keys(self):
        """获取可用API密钥列表"""
        return self.strategy.get_available_keys()

    def calculate_load_factor(self, api_key):
        """计算指定密钥的负载因子"""
        return self.strategy.calculate_load_factor(api_key)

    def get_load_status(self):
        """获取当前负载状态"""
        return self.strategy.get_capacity_info()

    def get_capacity_info(self):
        """获取容量信息"""
        current, max_cap = self.strategy.get_capacity_info()
        return {
            'type': self.strategy.capacity_type,
            'current': current,
            'max': max_cap,
            'weight': self.model_weight,
            'keys': [
                {
                    'key': k['key'],
                    'weight': k['weight'],
                    'current': self.get_key_load(k['key'])
                }
                for k in self.api_keys
            ]
        }

    def get_key_load(self, api_key):
        """获取单个密钥的当前负载"""
        if isinstance(self.strategy, ConcurrencyStrategy):
            return self.strategy.counters.get(api_key, 0)
        elif isinstance(self.strategy, QPSStrategy):
            return len(self.strategy.request_times.get(api_key, []))
        return 0

    @abstractmethod
    def chat_completion(self, messages, tools, api_key, stream=False):
        pass

    @property
    def model_weight(self):
        return self._model_weight

    @model_weight.setter
    def model_weight(self, value):
        if value <= 0:
            raise ValueError("Model weight must be positive")
        self._model_weight = float(value)