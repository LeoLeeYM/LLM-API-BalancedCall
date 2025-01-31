import time
import threading
from abc import ABC, abstractmethod
from collections import deque

class BaseLoadStrategy(ABC):
    def __init__(self, api_keys, **kwargs):
        self.api_keys = api_keys  # 格式: [{'key':..., 'weight':...}]
        self.locks = {k['key']: threading.Lock() for k in api_keys}
        self._init_params(**kwargs)

    @abstractmethod
    def _init_params(self, **kwargs):
        pass

    @property
    @abstractmethod
    def capacity_type(self):
        """返回容量类型标识"""
        pass

    @abstractmethod
    def track_request(self, api_key):
        pass

    @abstractmethod
    def release_request(self, api_key):
        pass

    @abstractmethod
    def get_available_keys(self):
        pass

    @abstractmethod
    def get_capacity_info(self):
        """返回 (当前负载, 最大容量)"""
        pass

class ConcurrencyStrategy(BaseLoadStrategy):
    def _init_params(self, max_concurrency):
        self.max_concurrency = max_concurrency
        self.counters = {k['key']: 0 for k in self.api_keys}

    @property
    def capacity_type(self):
        return "concurrency"

    def track_request(self, api_key):
        with self.locks[api_key]:
            if self.counters[api_key] < self.max_concurrency:
                self.counters[api_key] += 1
                return True
            return False

    def release_request(self, api_key):
        with self.locks[api_key]:
            self.counters[api_key] = max(0, self.counters[api_key] - 1)

    def get_available_keys(self):
        return [k for k in self.api_keys if self.counters[k['key']] < self.max_concurrency]

    def get_capacity_info(self):
        current = sum(self.counters.values())
        max_cap = len(self.api_keys) * self.max_concurrency
        return current, max_cap
    
    def calculate_load_factor(self, api_key):
        with self.locks[api_key]:
            return self.counters[api_key] / self.max_concurrency

class QPSStrategy(BaseLoadStrategy):
    def _init_params(self, max_qps):
        self.max_qps = max_qps
        self.request_times = {k['key']: deque() for k in self.api_keys}

    @property
    def capacity_type(self):
        return "qps"

    def track_request(self, api_key):
        with self.locks[api_key]:
            now = time.time()
            self._clean_old_requests(api_key, now)
            if len(self.request_times[api_key]) < self.max_qps:
                self.request_times[api_key].append(now)
                return True
            return False

    def _clean_old_requests(self, api_key, now):
        while self.request_times[api_key] and self.request_times[api_key][0] < now - 1:
            self.request_times[api_key].popleft()

    def release_request(self, api_key):
        pass  # QPS策略自动过期不需要释放

    def get_available_keys(self):
        now = time.time()
        available = []
        for key_info in self.api_keys:
            key = key_info['key']
            with self.locks[key]:
                self._clean_old_requests(key, now)
                if len(self.request_times[key]) < self.max_qps:
                    available.append(key_info)
        return available

    def get_capacity_info(self):
        current = sum(len(q) for q in self.request_times.values())
        max_cap = len(self.api_keys) * self.max_qps
        return current, max_cap
    
    def calculate_load_factor(self, api_key):
        with self.locks[api_key]:
            return len(self.request_times[api_key]) / self.max_qps