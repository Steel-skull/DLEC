import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.models.model_loader import ModelLoader
from src.analysis.activation_analyzer import ActivationAnalyzer
from src.models.model_configurator import ModelConfigurator
import logging
from src.utils.logger import setup_custom_logger

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze model activations and generate configurations.')
    parser.add_argument('--model', type=str, required=True, help='Model identifier on HuggingFace Models Hub')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset identifier on HuggingFace Datasets Hub')
    parser.add_argument('--dbpath', type=str, default='./activations.db', help='Path to the SQLite database for storing activations')
    return parser.parse_args()

def main():
    args = parse_arguments()
    logger = setup_custom_logger('main')
    logger.info(f"Loading model and tokenizer for '{args.model}'.")
    model_loader = ModelLoader(args.model)
    model, tokenizer = model_loader.get_model_and_tokenizer()
    logger.info(f"Loading dataset '{args.dataset}'.")
    dataset = load_dataset(args.dataset)
    analyzer = ActivationAnalyzer(model, tokenizer, args.dbpath)
    logger.info('Processing dataset and recording activations...')
    analyzer.process_dataset_and_record_activations(dataset)
    logger.info('Analyzing activations to identify beneficial layers...')
    beneficial_layers = analyzer.analyze_activations()
    logger.info(f'Beneficial layers identified: {beneficial_layers}')
    logger.info('Generating model configuration based on beneficial layers...')
    configurator = ModelConfigurator(beneficial_layers, args.model)
    config = configurator.generate_config()
    configurator.save_config_to_yaml(config, 'model_config.yaml')
    logger.info('Model configuration generated and saved.')

if __name__ == '__main__':
    main()
