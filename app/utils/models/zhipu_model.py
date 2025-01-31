from zhipuai import ZhipuAI
from .base_model import BaseModel
from app.utils.load_strategies import ConcurrencyStrategy

class ZhipuModel(BaseModel):
    STRATEGY_CLASS = ConcurrencyStrategy
    CONFIG_SECTION = 'ZHIPU_CONFIG'
    supports_tools = True

    def _get_strategy_params(self):
        return {'max_concurrency': self.config['max_concurrency']}

    def chat_completion(self, messages, tools, api_key, stream=False):
        client = ZhipuAI(api_key=api_key)
        try:
            if not self.strategy.track_request(api_key):
                raise RuntimeError("API key at capacity limit")
                
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=stream
            )
            
            if stream:
                return self._handle_stream(response, api_key)
            return response.choices[0].message.content
        finally:
            if not stream:
                self.strategy.release_request(api_key)

    def _handle_stream(self, response, api_key):
        try:
            for chunk in response:
                yield chunk.choices[0].delta.content or ""
        finally:
            self.strategy.release_request(api_key)