# LLM-API-BalancedCall
基于 Python Flask 的 LLM 模型 API 统一接口均衡调用系统，可快速添加任何 LLM 模型的 API 并设置对应的负载，系统会自动均衡负载，以实现免费 LLM API 的高效应用。
通过本项目你可以快速完成 LLM 模型 API 的聚合接口，并且模块化的设计允许您快速的添加任意模型 API 并设置其负载和权重。

项目结构

```
├── app/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── routes.py             // 内建 API 接口
│   └── utils/
│       ├── __init__.py
│       ├── llm_manager.py        // 模型调用类
│       ├── load_balancer.py      // 模型负载计算
|       ├── load_strategies.py    // 模型负载计算类型
│       └── models/
│           ├── __init__.py
│           ├── base_model.py     // 模型基础类
│           ├── zhipu_model.py    // 智谱 GLM-4-Flash 模型
├── config.py
├── wsgi.py
├── README.md
├── requirements.txt
└── LICENSE
```

## 目录

1. [快速上手指南](#1-快速上手指南)
2. [添加新模型教程](#2-添加新模型教程)
3. [高级配置：自定义负载策略](#3-高级配置自定义负载策略)
4. [生产部署建议](#4-生产部署建议)
5. [内建API接口文档](#5-内建API接口文档)

---

## 1. 快速上手指南

### 1.1 安装运行

以下是快速安装和运行本项目的步骤：

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
```

`config.py` 格式如下：

```python
class Config:
    # 智谱AI配置（带权重示例）
    ZHIPU_CONFIG = {   # 智谱模型配置
        'api_keys': [  # apiKey 集合
            {'key': 'your api key', 'weight': 1.0}  # 配置了 weight，影响选择该模型后 apiKey 的选择倾向
        ],
        'model_weight': 1, # 模型整体权重，影响模型选择倾向
        'max_concurrency': 200  # 模型负载参数
    }

    # 系统配置
    DEBUG = True
    FLASK_ENV = 'development'
    ENABLED_MODELS = ['zhipu']  # 注册的模型
```

```
# 启动服务
python run.py
```

### 1.2 验证服务

在启动服务后，你可以通过以下命令验证服务是否正常运行：

```bash
# 测试系统状态
curl http://localhost:9000/llm/system-capacity

# 发送示例请求
curl -X POST http://localhost:9000/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"你好"}]}'
```

------

## 2. 添加新模型教程

### 2.1 创建模型类

在 `app/utils/models/` 下新建文件（例如 `baidu_model.py`），并实现模型接口：

```python
import requests
from typing import Generator
from .base_model import BaseModel
from app.utils.load_strategies import ConcurrencyStrategy

class BaiduModel(BaseModel):
    """
    百度文心大模型实现
    文档参考：https://cloud.baidu.com/doc/WENXINWORKSHOP/
    """
    
    # 必须配置项 -----------
    STRATEGY_CLASS = ConcurrencyStrategy    # 选择负载策略，此处选择内置最大并发策略
    CONFIG_SECTION = 'BAIDU_CONFIG'         # 对应配置段名称
    supports_tools = False                  # 是否支持函数调用
    
    # API常量配置 ----------
    API_BASE = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
    DEFAULT_MODEL = "eb-instant"  # 默认模型版本
    
    def _get_strategy_params(self):
        """从配置获取策略参数"""
        return {
            'max_concurrency': self.config['max_concurrency']
        }

    def chat_completion(self, messages, tools, api_key, stream=False):
        """
        核心API调用方法
        :param messages: 消息列表
        :param tools: 工具列表（本示例不支持）
        :param api_key: 当前选用的API密钥
        :param stream: 是否流式传输
        :return: 同步返回字符串，流式返回生成器
        """
        # 构造请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # 构造请求体
        payload = {
            "messages": messages,
            "stream": stream
        }
        
        try:
            if stream:
                return self._handle_stream_request(headers, payload)
            return self._handle_sync_request(headers, payload)
        except Exception as e:
            self._handle_error(e)

    # 私有方法 -----------
    def _handle_sync_request(self, headers, payload):
        """处理同步请求"""
        response = requests.post(
            self.API_BASE,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['result']
    
    def _handle_stream_request(self, headers, payload) -> Generator[str, None, None]:
        """处理流式请求"""
        with requests.post(
            self.API_BASE,
            headers=headers,
            json=payload,
            stream=True,
            timeout=60
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    yield chunk.decode('utf-8')

    def _handle_error(self, error):
        """统一错误处理"""
        error_map = {
            requests.HTTPError: "API服务器返回错误",
            requests.Timeout: "请求超时",
            KeyError: "响应格式不符合预期"
        }
        raise error_map.get(type(error), "未知错误") from error
```

### 2.2 配置参数

修改 `config.py` 以配置 API 密钥和模型参数：

```python
class Config:
    # 百度文心配置
    BAIDU_CONFIG = {
        # API密钥配置（支持权重）
        'api_keys': [
            {'key': 'your_api_key_1', 'weight': 2.0},  # 高权重密钥
            {'key': 'your_api_key_2', 'weight': 1.0}   # 普通密钥
        ],
        
        # 模型参数
        'model_weight': 1.5,          # 模型全局权重
        'max_concurrency': 100,       # 单密钥最大并发
        
        # 可选高级参数
        'default_temperature': 0.7,   # 默认采样温度
        'max_retries': 3              # 请求重试次数
    }
    
    # 启用模型列表
    ENABLED_MODELS = ['zhipu', 'baidu']
```

#### 配置项说明

| 参数                | 类型  | 必需 | 说明                            |
| ------------------- | ----- | ---- | ------------------------------- |
| api_keys            | list  | 是   | API密钥列表，支持字典格式带权重 |
| model_weight        | float | 是   | 模型全局权重（>=1提升优先级）   |
| max_concurrency     | int   | 是   | 单密钥最大并发请求数            |
| default_temperature | float | 否   | 默认采样温度（0~1）             |
| max_retries         | int   | 否   | 请求失败重试次数                |

### 2.3 注册模型

#### 2.3.1 修改注册文件

编辑 `app/utils/models/__init__.py` 注册模型：

```python
from .baidu_model import BaiduModel

MODEL_CLASSES = {
    'baidu': BaiduModel,  # 键名对应ENABLED_MODELS中的名称
    # 其他模型...
}
```

#### 2.3.2 注册验证

测试注册是否成功：

```python
# 测试注册是否成功
from app.utils.models import MODEL_CLASSES
print(MODEL_CLASSES['baidu'])  # 应输出 <class 'app.utils.models.baidu_model.BaiduModel'>
```

### 2.4 实现API调用

#### 2.4.1 请求体构建

根据模型要求调整 `_build_payload` 方法：

```python
def _build_payload(self, messages, stream=False):
    return {
        "messages": messages,
        "temperature": self.config.get('default_temperature', 0.7),
        "stream": stream,
        # 添加模型特定参数
        "disable_search": False  # 示例：百度文心特有参数
    }
```

#### 2.4.2 响应解析

```python
def _parse_response(self, response_data):
    """解析同步响应"""
    try:
        return response_data['result']
    except KeyError:
        raise ValueError("Invalid response format")
```

### 2.5 负载策略配置

#### 2.5.1 策略选择对照表

| 模型特性   | 推荐策略            | 配置示例             |
| ---------- | ------------------- | -------------------- |
| 限制并发数 | ConcurrencyStrategy | max_concurrency: 100 |
| 限制QPS    | QPSStrategy         | max_qps: 5           |
| 混合限制   | 自定义策略          | 继承BaseLoadStrategy |

#### 2.5.2 权重配置示例

```python
# 密钥权重影响流量分配比例
'api_keys': [
    {'key': 'pro_key', 'weight': 3.0},  # 60%流量 (3/(3+2))
    {'key': 'basic_key', 'weight': 2.0} # 40%流量
]

# 模型权重影响跨模型选择
BAIDU_CONFIG = {'model_weight': 1.5}  # 比默认模型多50%流量
```

## 3. 高级配置：自定义负载策略

### 3.1 创建策略类

在 `app/utils/load_strategies.py` 中添加自定义负载策略：

```python
class ResponseTimeStrategy(BaseLoadStrategy):
    """
    基于响应时间的动态负载策略
    根据历史响应时间自动调整权重
    """
    
    def _init_params(self, max_concurrency, decay_factor=0.9):
        self.max_concurrency = max_concurrency
        self.decay_factor = decay_factor  # 历史衰减因子
        self.counters = {k['key']: 0 for k in self.api_keys}
        self.response_times = {k['key']: deque(maxlen=100) for k in self.api_keys}
        
    @property
    def capacity_type(self):
        return "concurrency"

    def track_request(self, api_key):
        with self.locks[api_key]:
            if self.counters[api_key] < self.max_concurrency:
                self.counters[api_key] += 1
                return True
            return False

    def record_response_time(self, api_key, response_time):
        """记录响应时间（需在模型类中调用）"""
        with self.locks[api_key]:
            self.response_times[api_key].append(response_time)

    def calculate_load_factor(self, api_key):
        with self.locks[api_key]:
            # 计算指数加权平均响应时间
            times = list(self.response_times[api_key])
            if not times:
                return 0
            avg_time = times[0]
            for t in times[1:]:
                avg_time = self.decay_factor * avg_time + (1-self.decay_factor)*t
            return avg_time * self.counters[api_key]

    def get_available_keys(self):
        return [k for k in self.api_keys if self.counters[k['key']] < self.max_concurrency]

    def get_capacity_info(self):
        current = sum(self.counters.values())
        return current, len(self.api_keys)*self.max_concurrency
```

### 3.2 在模型中使用

修改模型类来使用自定义策略：

```python
class CustomModel(BaseModel):
    STRATEGY_CLASS = ResponseTimeStrategy  # 使用自定义策略
    
    def chat_completion(self, ...):
        start_time = time.time()
        try:
            # ...执行请求...
        finally:
            response_time = time.time() - start_time
            self.strategy.record_response_time(api_key, response_time)
```

### 3.3 配置参数

```python
CUSTOM_CONFIG = {
    'api_keys': [...],
    'max_concurrency': 50,
    'decay_factor': 0.85  # 历史衰减系数
}
```

------

## 4. 生产部署建议

### 4.1 性能优化

使用连接池优化请求性能：

```python
from requests.adapters import HTTPAdapter

class CustomModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.session = requests.Session()
        adapter = HTTPAdapter(
            pool_connections=100,
            pool_maxsize=100,
            max_retries=3
        )
        self.session.mount('https://', adapter)
```

### 4.2 监控配置

推荐使用 Prometheus 进行监控，以下是指标示例：

```python
# Prometheus 指标示例
llm_requests_total{model="custom", status="success"} 238
llm_requests_duration_seconds{quantile="0.95"} 0.23
llm_load_factor{model="custom"} 0.65
```

### 4.3 自动扩缩容

```python
# 自动调整密钥权重
def auto_scale_weights():
    for model in models.values():
        for key_info in model.api_keys:
            load = model.get_key_load(key_info['key'])
            # 根据负载动态调整权重
            key_info['weight'] = 1.0 / (load + 0.1)  # 负载越高权重越低
```

------

## 5. 内建API接口文档

### 1. 普通对话接口

**端点**: POST /llm/chat

**功能**: 发送普通对话请求，返回完整响应。

**请求示例**:

```bash
curl -X POST http://localhost:9000/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
      "messages": [
          {"role": "user", "content": "你好，请介绍一下你自己"}
      ]
  }'
```

**请求体**:

```json
{
    "messages": [
        {"role": "user", "content": "你好，请介绍一下你自己"}
    ],
    "tools": []  // 可选，函数调用工具列表
}
```

**响应示例**:

```json
{
    "result": "你好！我是一个人工智能助手，专门设计用来回答各种问题和提供帮助。"
}
```

### 2. 流式对话接口

**端点**: POST /llm/chat/stream

**功能**: 发送流式对话请求，逐块返回响应。

**请求示例**:

```bash
curl -X POST http://localhost:9000/llm/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
      "messages": [
          {"role": "user", "content": "用Python写一个快速排序算法"}
      ]
  }'
```

**请求体**:

```json
{
    "messages": [
        {"role": "user", "content": "用Python写一个快速排序算法"}
    ],
    "tools": []  // 可选，函数调用工具列表
}
```

**响应示例**:

```
快
速
排
序
算
法
...
```

------

### 3. 系统负载查询接口

**端点**: `GET /llm/system-load`

**功能**: 获取当前系统的负载率（百分比）。

**请求示例**:

```bash
curl http://localhost:9000/llm/system-load
```

**响应示例**:

```json
{
    "load_percent": 34.56
}
```

------

### 4. 系统容量查询接口

**端点**: `GET /llm/system-capacity`

**功能**: 获取系统的理论最大容量和当前负载详情。

**请求示例**:

```bash
curl http://localhost:9000/llm/system-capacity
```

**响应示例**:

```json
{
    "total_max": {
        "concurrency": 400,
        "qps": 10
    },
    "details": {
        "zhipu": {
            "type": "concurrency",
            "current": 123,
            "max": 200,
            "weight": 1.2,
            "keys": [
                {
                    "key": "key1",
                    "weight": 2.0,
                    "current": 80
                },
                {
                    "key": "key2",
                    "weight": 1.0,
                    "current": 43
                }
            ]
        },
        "spark": {
            "type": "qps",
            "current": 3,
            "max": 4,
            "weight": 1.0,
            "keys": [
                {
                    "key": "keyA",
                    "weight": 1.0,
                    "current": 2
                },
                {
                    "key": "keyB",
                    "weight": 1.0,
                    "current": 1
                }
            ]
        }
    }
}
```

------

### 5. 模型负载详情接口

默认不启用的接口，该接口会**暴露 APIKey**，如需启用请手动删除 `app/llm/routes.py` 中该接口的注释

**端点**: `GET /llm/model-load/<model_name>`

**功能**: 获取指定模型的负载详情。

**请求示例**:

```bash
curl http://localhost:9000/llm/model-load/zhipu
```

**响应示例**:

```json
{
    "type": "concurrency",
    "current": 123,
    "max": 200,
    "weight": 1.2,
    "keys": [
        {
            "key": "key1",
            "weight": 2.0,
            "current": 80
        },
        {
            "key": "key2",
            "weight": 1.0,
            "current": 43
        }
    ]
}
```

------

### 6. 密钥负载详情接口

**端点**: `GET /llm/key-load/<model_name>/<api_key>`

**功能**: 获取指定模型和密钥的当前负载。

**请求示例**:

```bash
curl http://localhost:9000/llm/key-load/zhipu/key1
```

**响应示例**:

```json
{
    "key": "key1",
    "weight": 2.0,
    "current": 80,
    "max": 100
}
```

------

### 8. 健康检查接口

**端点**: `GET /llm/health`

**功能**: 检查服务是否正常运行。

**请求示例**:

```bash
curl http://localhost:9000/health
```

**响应示例**:

```json
{
    "status": "ok",
    "timestamp": "2023-10-15T12:34:56Z"
}
```

------

### 9. 模型列表接口

默认不启用的接口，该接口会**暴露 APIKey**，如需启用请手动删除 `app/llm/routes.py` 中该接口的注释

**端点**: `GET /llm/models`

**功能**: 获取当前启用的模型列表。

**请求示例**:

```bash
curl http://localhost:9000/llm/models
```

**响应示例**:

```json
{
    "models": ["zhipu", "spark", "custom"]
}
```

------

### 10. 密钥列表接口

默认不启用的接口，该接口会**暴露 APIKey**，如需启用请手动删除 `app/llm/routes.py` 中该接口的注释

**端点**: `GET /llm/keys/<model_name>`

**功能**: 获取指定模型的 API 密钥列表。

**请求示例**:

```bash
curl http://localhost:9000/llm/keys/zhipu
```

**响应示例**:

```json
{
    "keys": [
        {"key": "key1", "weight": 2.0},
        {"key": "key2", "weight": 1.0}
    ]
}
```



遇到问题请参考代码注释或提交Issue讨论。
