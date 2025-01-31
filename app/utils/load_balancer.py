class LoadBalancer:
    def select_instance(self, models, requires_tools=False):
        candidates = []
        
        for model_name, model in models.items():
            # 检查工具支持
            if requires_tools and not model.supports_tools:
                continue
                
            # 获取模型级权重
            model_weight = model.model_weight
            
            # 遍历可用密钥
            for key_info in model.get_available_keys():
                key = key_info['key']
                key_weight = key_info['weight']
                
                # 计算综合权重
                combined_weight = model_weight * key_weight
                
                # 计算调整后的负载因子
                load_factor = model.calculate_load_factor(key) / combined_weight
                
                candidates.append({
                    'score': load_factor,
                    'model': model,
                    'key_info': key_info
                })
        
        if not candidates:
            raise RuntimeError("No available instances")
        
        # 选择最优实例
        best = min(candidates, key=lambda x: x['score'])
        return best['model'], best['key_info']['key']