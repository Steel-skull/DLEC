import yaml
import logging
logger = logging.getLogger(__name__)

class ModelConfigurator:

    def __init__(self, beneficial_layers, model_name):
        self.beneficial_layers = beneficial_layers
        self.model_name = model_name

    def generate_config(self):
        config = {'dtype': 'bfloat16', 'merge_method': 'passthrough', 'slices': []}
        step = 4
        for i in range(0, len(self.beneficial_layers), step):
            slice_layers = self.beneficial_layers[i:i + step]
            if slice_layers:
                slice_config = {'sources': [{'model': self.model_name, 'layer_range': [slice_layers[0], slice_layers[-1]]}]}
                config['slices'].append(slice_config)
        return config

    def save_config_to_yaml(self, config, file_path='model_config.yaml'):
        try:
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f'Configuration saved to {file_path}.')
        except Exception as e:
            logger.error(f'Failed to save configuration to {file_path}: {e}')
