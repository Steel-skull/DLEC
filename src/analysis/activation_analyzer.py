import torch
from tqdm import tqdm
import logging
from src.database.database_manager import DatabaseManager
from src.analysis.analyze_layer import analyze_layer_helper
from multiprocessing import Pool
logger = logging.getLogger(__name__)

class ActivationAnalyzer:

    def __init__(self, model, tokenizer, db_path, chunk_size=1000, num_bins=100):
        self.model = model
        self.tokenizer = tokenizer
        self.db_manager = DatabaseManager(db_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.chunk_size = chunk_size
        self.num_bins = num_bins

    def process_dataset_and_record_activations(self, dataset):
        self.db_manager.setup_database()
        activations = []
        total_examples = len(dataset['train'])
        progress_bar = tqdm(total=total_examples, desc='Processing dataset', unit='example')
        for example in dataset['train']:
            conversation = example['conversations']
            text = ' '.join((turn['value'] for turn in conversation))
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states
            for layer_idx, layer_activations in enumerate(hidden_states):
                for neuron_idx in range(layer_activations.size(2)):
                    activation = layer_activations[0, 0, neuron_idx].item()
                    activations.append((f'layer_{layer_idx}', neuron_idx, activation))
            if len(activations) >= 10000:
                self.db_manager.record_activations_to_db(activations)
                activations = []
            progress_bar.update(1)
        if activations:
            self.db_manager.record_activations_to_db(activations)
        progress_bar.close()

    def analyze_activations(self):
        layers = [f'layer_{i}' for i in range(self.model.config.num_hidden_layers)]
        args_list = [(layer, self.chunk_size, self.num_bins, self.device, self.db_manager.db_path) for layer in layers]
        with Pool() as pool:
            results = pool.map(analyze_layer_helper, args_list)
        beneficial_layers = [result[1] for result in results if result[1] is not None]
        return beneficial_layers
