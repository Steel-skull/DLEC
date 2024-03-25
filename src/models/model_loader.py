import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model_and_tokenizer(self):
        """Loads the specified model and tokenizer, setting up quantization if applicable."""
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto"
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"Model and tokenizer loaded successfully: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading model and tokenizer: {e}")

    def get_model_and_tokenizer(self):
        """Returns the loaded model and tokenizer."""
        if not self.model or not self.tokenizer:
            self.load_model_and_tokenizer()
        return self.model, self.tokenizer
