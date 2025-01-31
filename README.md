# LLM-API-BalancedCall
基于 Python Flask 的 LLM 模型 API 统一接口均衡调用系统，可快速添加任何 LLM 模型的 API 并设置对应的负载，系统会自动均衡负载，以实现免费 LLM API 的高效应用。
通过本项目你可以快速完成 LLM 模型 API 的聚合接口，并且模块化的设计允许您快速的添加任意模型 API 并设置其负载和权重。

## 目录

1. [快速上手指南](#1-快速上手指南)
2. [添加新模型教程](#2-添加新模型教程)
3. [高级配置：自定义负载策略](#3-高级配置自定义负载策略)
4. [生产部署建议](#4-生产部署建议)

---

## 1. 快速上手指南

### 1.1 安装运行

```bash
# 克隆仓库
git clone https://github.com/LeoLeeYM/LLM-API-BalancedCall.git
cd LLM-API-BalancedCall

# 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 配置密钥（编辑config.py）
nano config.py

# 启动服务
python run.py
```

### 1.2 验证服务

```bash
# 测试系统状态
curl http://localhost:9000/llm/system-capacity

# 发送示例请求
curl -X POST http://localhost:9000/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

## 2. 添加新模型教程

### 2.1 创建模型类

在 `app/utils/models/` 下新建 `custom_model.py`：
```python
import requests
from .base_model import BaseModel
from app.utils.load_strategies import QPSStrategy

class CustomModel(BaseModel):
    """
    自定义大模型实现类
    
    该类继承自 BaseModel，用于实现与自定义大模型 API 的交互。
    支持同步和流式请求，使用 QPS 策略进行负载均衡。
    """
    
    # 基本配置
    STRATEGY_CLASS = QPSStrategy  # 使用 QPS 策略进行负载均衡
    CONFIG_SECTION = 'CUSTOM_CONFIG'  # 对应配置文件中的配置段名称
    supports_tools = False  # 是否支持工具调用（函数调用）
    API_ENDPOINT = "https://api.custom-model.com/v1/chat"  # API 请求地址

    def _get_strategy_params(self):
        """
        获取负载策略所需的参数
        
        从配置中读取 QPS 限制，并返回给策略类使用。
        """
        return {'max_qps': self.config['qps_limit']}

    def chat_completion(self, messages, tools, api_key, stream=False):
        """
        核心方法：执行对话请求
        
        :param messages: 消息列表，格式为 [{"role": "user", "content": "你好"}]
        :param tools: 工具列表（本示例不支持）
        :param api_key: 当前使用的 API 密钥
        :param stream: 是否使用流式传输
        :return: 如果 stream=True，返回生成器；否则返回字符串
        """
        # 构造请求头
        headers = {"Authorization": f"Bearer {api_key}"}
        
        # 构造请求体
        payload = self._build_payload(messages, stream)
        
        try:
            # 发送 POST 请求
            response = requests.post(
                self.API_ENDPOINT,
                headers=headers,
                json=payload,
                stream=stream,  # 是否流式传输
                timeout=30  # 请求超时时间
            )
            response.raise_for_status()  # 检查 HTTP 状态码，非 2xx 会抛出异常
            
            # 解析响应
            return self._parse_response(response, stream)
        except Exception as e:
            # 统一错误处理
            self._handle_error(e)

    # 私有方法 -----------

    def _build_payload(self, messages, stream):
        """
        构造请求体
        
        :param messages: 消息列表
        :param stream: 是否流式传输
        :return: 构造好的请求体字典
        """
        return {
            "messages": messages,  # 对话消息
            "temperature": 0.7,  # 采样温度，控制生成多样性
            "stream": stream  # 是否流式传输
        }

    def _parse_response(self, response, stream):
        """
        解析 API 响应
        
        :param response: requests 响应对象
        :param stream: 是否流式传输
        :return: 如果 stream=True，返回生成器；否则返回解析后的字符串
        """
        if stream:
            # 流式响应：逐块返回数据
            for chunk in response.iter_content():
                yield chunk.decode()  # 解码字节流为字符串
        else:
            # 同步响应：直接返回完整结果
            return response.json()['choices'][0]['message']['content']

    def _handle_error(self, error):
        """
        统一错误处理方法
        
        :param error: 捕获的异常对象
        :raises: 转换后的业务异常
        """
        # 错误类型映射
        error_map = {
            requests.HTTPError: "API服务器返回错误",  # HTTP 状态码异常
            requests.Timeout: "请求超时"  # 请求超时
        }
        # 根据错误类型抛出对应的业务异常
        raise error_map.get(type(error), "未知错误") from error
```

### 2.2 配置参数
修改 `config.py`：

```python
class Config:
    CUSTOM_CONFIG = {
        'api_keys': [
            {'key': 'your_key_1', 'weight': 1.5},
            {'key': 'your_key_2', 'weight': 1.0}
        ],
        'model_weight': 1.2,
        'qps_limit': 5,
        'timeout': 30
    }
    
    ENABLED_MODELS = ['zhipu', 'custom']
```

### 2.3 注册模型

修改 `app/utils/models/__init__.py`：

```python
from .custom_model import CustomModel

MODEL_CLASSES = {
    'custom': CustomModel,
    # 其他模型...
}

```

### 2.4 测试验证

```python
# tests/test_custom_model.py
import unittest
from app.utils.models.custom_model import CustomModel
from config import Config

class TestCustomModel(unittest.TestCase):
    def setUp(self):
        self.model = CustomModel(Config)
        self.valid_key = Config.CUSTOM_CONFIG['api_keys'][0]['key']
    
    def test_sync_chat(self):
        result = self.model.chat_completion(
            [{"role":"user","content":"你好"}],
            None,
            self.valid_key
        )
        self.assertIsInstance(result, str)

if __name__ == '__main__':
    unittest.main()
```
