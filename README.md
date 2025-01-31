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
    """自定义大模型实现"""
    
    # 基本配置
    STRATEGY_CLASS = QPSStrategy
    CONFIG_SECTION = 'CUSTOM_CONFIG'
    supports_tools = False
    API_ENDPOINT = "https://api.custom-model.com/v1/chat"

    def _get_strategy_params(self):
        return {'max_qps': self.config['qps_limit']}

    def chat_completion(self, messages, tools, api_key, stream=False):
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = self._build_payload(messages, stream)
        
        try:
            response = requests.post(
                self.API_ENDPOINT,
                headers=headers,
                json=payload,
                stream=stream,
                timeout=30
            )
            response.raise_for_status()
            
            return self._parse_response(response, stream)
        except Exception as e:
            self._handle_error(e)

    # 私有方法
    def _build_payload(self, messages, stream):
        return {
            "messages": messages,
            "temperature": 0.7,
            "stream": stream
        }

    def _parse_response(self, response, stream):
        if stream:
            for chunk in response.iter_content():
                yield chunk.decode()
        else:
            return response.json()['choices'][0]['message']['content']

    def _handle_error(self, error):
        error_map = {
            requests.HTTPError: "API服务器返回错误",
            requests.Timeout: "请求超时"
        }
        raise error_map.get(type(error), "未知错误") from error
```

