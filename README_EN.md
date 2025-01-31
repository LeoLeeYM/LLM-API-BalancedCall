# LLM-API-BalancedCall
A Python Flask-based unified API interface for balanced calling of LLM models. This system allows you to quickly integrate any LLM model API and set corresponding load weights. The system automatically balances the load to achieve efficient utilization of free LLM APIs.  
With this project, you can rapidly create an aggregated interface for LLM model APIs. The modular design enables you to quickly add any model API and configure its load and weight.

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── routes.py             // Built-in API endpoints
│   └── utils/
│       ├── __init__.py
│       ├── llm_manager.py        // Model invocation class
│       ├── load_balancer.py      // Model load balancing
|       ├── load_strategies.py    // Load balancing strategies
│       └── models/
│           ├── __init__.py
│           ├── base_model.py     // Base model class
│           ├── zhipu_model.py    // Zhipu GLM-4-Flash model
├── config.py
├── wsgi.py
├── README.md
├── requirements.txt
└── LICENSE
```

## Table of Contents

1. [Quick Start Guide](#1-quick-start-guide)
2. [Adding a New Model Tutorial](#2-adding-a-new-model-tutorial)
3. [Advanced Configuration: Custom Load Balancing Strategies](#3-advanced-configuration-custom-load-balancing-strategies)
4. [Production Deployment Recommendations](#4-production-deployment-recommendations)
5. [Built-in API Documentation](#5-built-in-api-documentation)

---

## 1. Quick Start Guide

### 1.1 Installation and Setup

Follow these steps to quickly install and run the project:

```bash
# Clone the repository
git clone https://github.com/LeoLeeYM/LLM-API-BalancedCall.git
cd LLM-API-BalancedCall

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Configure API keys (edit config.py)
nano config.py
```

The `config.py` file should follow this format:

```python
class Config:
    # Zhipu AI configuration (with weight example)
    ZHIPU_CONFIG = {   # Zhipu model configuration
        'api_keys': [  # API key collection
            {'key': 'your api key', 'weight': 1.0}  # Weight affects the selection tendency of this API key
        ],
        'model_weight': 1, # Global model weight, affects model selection tendency
        'max_concurrency': 200  # Model load parameter
    }

    # System configuration
    DEBUG = True
    FLASK_ENV = 'development'
    ENABLED_MODELS = ['zhipu']  # Registered models
```

```
# Start the service
python run.py
```

### 1.2 Verify the Service

After starting the service, you can verify its operation with the following commands:

```bash
# Test system status
curl http://localhost:9000/llm/system-capacity

# Send a sample request
curl -X POST http://localhost:9000/llm/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

------

## 2. Adding a New Model Tutorial

### 2.1 Create a Model Class

Create a new file in `app/utils/models/` (e.g., `baidu_model.py`) and implement the model interface:

```python
import requests
from typing import Generator
from .base_model import BaseModel
from app.utils.load_strategies import ConcurrencyStrategy

class BaiduModel(BaseModel):
    """
    Implementation of Baidu's ERNIE model
    Documentation: https://cloud.baidu.com/doc/WENXINWORKSHOP/
    """
    
    # Required configurations -----------
    STRATEGY_CLASS = ConcurrencyStrategy    # Load balancing strategy, using built-in max concurrency strategy
    CONFIG_SECTION = 'BAIDU_CONFIG'         # Corresponding configuration section name
    supports_tools = False                  # Whether function calling is supported
    
    # API constants ----------
    API_BASE = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat"
    DEFAULT_MODEL = "eb-instant"  # Default model version
    
    def _get_strategy_params(self):
        """Get strategy parameters from configuration"""
        return {
            'max_concurrency': self.config['max_concurrency']
        }

    def chat_completion(self, messages, tools, api_key, stream=False):
        """
        Core API call method
        :param messages: List of messages
        :param tools: List of tools (not supported in this example)
        :param api_key: Selected API key
        :param stream: Whether to stream the response
        :return: Synchronous response as string, streaming response as generator
        """
        # Construct request headers
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Construct request payload
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

    # Private methods -----------
    def _handle_sync_request(self, headers, payload):
        """Handle synchronous requests"""
        response = requests.post(
            self.API_BASE,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['result']
    
    def _handle_stream_request(self, headers, payload) -> Generator[str, None, None]:
        """Handle streaming requests"""
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
        """Unified error handling"""
        error_map = {
            requests.HTTPError: "API server returned an error",
            requests.Timeout: "Request timed out",
            KeyError: "Unexpected response format"
        }
        raise error_map.get(type(error), "Unknown error") from error
```

### 2.2 Configure Parameters

Modify `config.py` to configure API keys and model parameters:

```python
class Config:
    # Baidu ERNIE configuration
    BAIDU_CONFIG = {
        # API key configuration (supports weights)
        'api_keys': [
            {'key': 'your_api_key_1', 'weight': 2.0},  # High-weight key
            {'key': 'your_api_key_2', 'weight': 1.0}   # Normal key
        ],
        
        # Model parameters
        'model_weight': 1.5,          # Global model weight
        'max_concurrency': 100,       # Max concurrency per key
        
        # Optional advanced parameters
        'default_temperature': 0.7,   # Default sampling temperature
        'max_retries': 3              # Request retry count
    }
    
    # Enabled models list
    ENABLED_MODELS = ['zhipu', 'baidu']
```

#### Configuration Details

| Parameter           | Type  | Required | Description                                      |
| ------------------- | ----- | -------- | ------------------------------------------------ |
| api_keys            | list  | Yes      | List of API keys, supports weighted dictionaries |
| model_weight        | float | No       | Global model weight (>=1 increases priority)     |
| max_concurrency     | int   | No       | Max concurrent requests per key                  |
| default_temperature | float | No       | Default sampling temperature (0~1)               |
| max_retries         | int   | No       | Number of retries on request failure             |

### 2.3 Register the Model

#### 2.3.1 Modify Registration File

Edit `app/utils/models/__init__.py` to register the model:

```python
from .baidu_model import BaiduModel

MODEL_CLASSES = {
    'baidu': BaiduModel,  # Key corresponds to the name in ENABLED_MODELS
    # Other models...
}
```

#### 2.3.2 Verify Registration

Test if the registration is successful:

```python
# Test registration
from app.utils.models import MODEL_CLASSES
print(MODEL_CLASSES['baidu'])  # Should output <class 'app.utils.models.baidu_model.BaiduModel'>
```

### 2.4 Implement API Calls

#### 2.4.1 Build Request Payload

Adjust the `_build_payload` method according to the model's requirements:

```python
def _build_payload(self, messages, stream=False):
    return {
        "messages": messages,
        "temperature": self.config.get('default_temperature', 0.7),
        "stream": stream,
        # Add model-specific parameters
        "disable_search": False  # Example: Baidu-specific parameter
    }
```

#### 2.4.2 Parse Response

```python
def _parse_response(self, response_data):
    """Parse synchronous response"""
    try:
        return response_data['result']
    except KeyError:
        raise ValueError("Invalid response format")
```

### 2.5 Load Balancing Strategy Configuration

#### 2.5.1 Strategy Selection Reference

| Model Characteristics | Strategy                                                     | Configuration Example    |
| --------------------- | ------------------------------------------------------------ | ------------------------ |
| Concurrency limit     | ConcurrencyStrategy                                          | max_concurrency: 100     |
| QPS limit             | QPSStrategy                                                  | max_qps: 5               |
| Mixed limits          | [Custom Strategy](#3-advanced-configuration-custom-load-balancing-strategies) | Inherit BaseLoadStrategy |

#### 2.5.2 Weight Configuration Example

```python
# Key weights affect traffic distribution
'api_keys': [
    {'key': 'pro_key', 'weight': 3.0},  # 60% traffic (3/(3+2))
    {'key': 'basic_key', 'weight': 2.0} # 40% traffic
]

# Model weights affect cross-model selection
BAIDU_CONFIG = {'model_weight': 1.5}  # 50% more traffic than default models
```

## 3. Advanced Configuration: Custom Load Balancing Strategies

### 3.1 Create a Strategy Class

Add a custom load balancing strategy in `app/utils/load_strategies.py`:

```python
class ResponseTimeStrategy(BaseLoadStrategy):
    """
    Dynamic load balancing strategy based on response time
    Automatically adjusts weights based on historical response times
    """
    
    def _init_params(self, max_concurrency, decay_factor=0.9):
        self.max_concurrency = max_concurrency
        self.decay_factor = decay_factor  # Historical decay factor
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
        """Record response time (to be called in the model class)"""
        with self.locks[api_key]:
            self.response_times[api_key].append(response_time)

    def calculate_load_factor(self, api_key):
        with self.locks[api_key]:
            # Calculate exponential weighted average response time
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

### 3.2 Use in Model Class

Modify the model class to use the custom strategy:

```python
class CustomModel(BaseModel):
    STRATEGY_CLASS = ResponseTimeStrategy  # Use custom strategy
    
    def chat_completion(self, ...):
        start_time = time.time()
        try:
            # ...Execute request...
        finally:
            response_time = time.time() - start_time
            self.strategy.record_response_time(api_key, response_time)
```

### 3.3 Configure Parameters

```python
CUSTOM_CONFIG = {
    'api_keys': [...],
    'max_concurrency': 50,
    'decay_factor': 0.85  # Historical decay factor
}
```

------

## 4. Production Deployment Recommendations

### 4.1 Performance Optimization

Use connection pooling to optimize request performance:

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

### 4.2 Monitoring Configuration

Recommended to use Prometheus for monitoring. Example metrics:

```python
# Prometheus metrics example
llm_requests_total{model="custom", status="success"} 238
llm_requests_duration_seconds{quantile="0.95"} 0.23
llm_load_factor{model="custom"} 0.65
```

### 4.3 Auto-scaling

```python
# Automatically adjust key weights
def auto_scale_weights():
    for model in models.values():
        for key_info in model.api_keys:
            load = model.get_key_load(key_info['key'])
            # Dynamically adjust weights based on load
            key_info['weight'] = 1.0 / (load + 0.1)  # Higher load, lower weight
```

------

## 5. Built-in API Documentation

### 1. Standard Chat Interface

**Endpoint**: POST /llm/chat

**Function**: Send a standard chat request and receive a complete response.

**Request Example**:

```bash
curl -X POST http://localhost:9000/llm/chat \
  -H "Content-Type: application/json" \
  -d '{
      "messages": [
          {"role": "user", "content": "Hello, please introduce yourself"}
      ]
  }'
```

**Request Body**:

```json
{
    "messages": [
        {"role": "user", "content": "Hello, please introduce yourself"}
    ],
    "tools": []  // Optional, list of function calling tools
}
```

**Response Example**:

```json
{
    "result": "Hello! I am an AI assistant designed to answer questions and provide help."
}
```

### 2. Streaming Chat Interface

**Endpoint**: POST /llm/chat/stream

**Function**: Send a streaming chat request and receive responses in chunks.

**Request Example**:

```bash
curl -X POST http://localhost:9000/llm/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
      "messages": [
          {"role": "user", "content": "Write a quicksort algorithm in Python"}
      ]
  }'
```

**Request Body**:

```json
{
    "messages": [
        {"role": "user", "content": "Write a quicksort algorithm in Python"}
    ],
    "tools": []  // Optional, list of function calling tools
}
```

**Response Example**:

```
Quick
sort
algorithm
...
```

------

### 3. System Load Query Interface

**Endpoint**: `GET /llm/system-load`

**Function**: Get the current system load percentage.

**Request Example**:

```bash
curl http://localhost:9000/llm/system-load
```

**Response Example**:

```json
{
    "load_percent": 34.56
}
```

------

### 4. System Capacity Query Interface

**Endpoint**: `GET /llm/system-capacity`

**Function**: Get the theoretical maximum capacity and current load details of the system.

**Request Example**:

```bash
curl http://localhost:9000/llm/system-capacity
```

**Response Example**:

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

### 5. Model Load Details Interface

This interface is disabled by default as it **exposes API keys**. To enable, manually uncomment the endpoint in `app/llm/routes.py`.

**Endpoint**: `GET /llm/model-load/<model_name>`

**Function**: Get load details for a specified model.

**Request Example**:

```bash
curl http://localhost:9000/llm/model-load/zhipu
```

**Response Example**:

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

### 6. Key Load Details Interface

**Endpoint**: `GET /llm/key-load/<model_name>/<api_key>`

**Function**: Get the current load for a specified model and API key.

**Request Example**:

```bash
curl http://localhost:9000/llm/key-load/zhipu/key1
```

**Response Example**:

```json
{
    "key": "key1",
    "weight": 2.0,
    "current": 80,
    "max": 100
}
```

------

### 8. Health Check Interface

**Endpoint**: `GET /llm/health`

**Function**: Check if the service is running normally.

**Request Example**:

```bash
curl http://localhost:9000/health
```

**Response Example**:

```json
{
    "status": "ok",
    "timestamp": "2023-10-15T12:34:56Z"
}
```

------

### 9. Model List Interface

This interface is disabled by default as it **exposes API keys**. To enable, manually uncomment the endpoint in `app/llm/routes.py`.

**Endpoint**: `GET /llm/models`

**Function**: Get the list of currently enabled models.

**Request Example**:

```bash
curl http://localhost:9000/llm/models
```

**Response Example**:

```json
{
    "models": ["zhipu", "spark", "custom"]
}
```

------

### 10. Key List Interface

This interface is disabled by default as it **exposes API keys**. To enable, manually uncomment the endpoint in `app/llm/routes.py`.

**Endpoint**: `GET /llm/keys/<model_name>`

**Function**: Get the list of API keys for a specified model.

**Request Example**:

```bash
curl http://localhost:9000/llm/keys/zhipu
```

**Response Example**:

```json
{
    "keys": [
        {"key": "key1", "weight": 2.0},
        {"key": "key2", "weight": 1.0}
    ]
}
```
