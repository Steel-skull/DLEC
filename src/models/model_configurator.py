import yaml
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ModelConfigurator:
    def __init__(self, beneficial_layers, model_name):
        self.beneficial_layers = beneficial_layers
        self.model_name = model_name

    def generate_config(self):
        """Generates a model configuration based on the analysis of beneficial layers."""
        config = {
            "dtype": "bfloat16",
            "merge_method": "passthrough",
            "slices": []
        }

        # Ensure step is appropriate for the expected batching of layer ranges.
        step = 4
        for i in range(0, len(self.beneficial_layers), step):
            slice_layers = self.beneficial_layers[i:i+step]  # Correctly slice the beneficial_layers list

            if slice_layers:  # Check if the slice contains any layers
                slice_config = {
                    "sources": [{
                        "model": self.model_name,
                        # Adjusted to use the actual layer numbers from slice_layers
                        "layer_range": [slice_layers[0], slice_layers[-1]]
                    }]
                }
                config["slices"].append(slice_config)

        return config

    def save_config_to_yaml(self, config, file_path='model_config.yaml'):
        """Saves the generated configuration to a YAML file.

        Args:
            config (dict): The configuration dictionary to be saved.
            file_path (str): The file path for saving the YAML configuration.
        """
        try:
            with open(file_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.info(f"Configuration saved to {file_path}.")
        except Exception as e:
            logger.error(f"Failed to save configuration to {file_path}: {e}")
