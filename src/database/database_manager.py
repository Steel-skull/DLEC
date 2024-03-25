import sqlite3
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path

    def setup_database(self):
        """Initializes the database and creates the necessary tables if they do not exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''CREATE TABLE IF NOT EXISTS activations (
                                  layer TEXT,
                                  neuron INTEGER,
                                  activation REAL)''')
                conn.commit()
                logger.info("Database setup successfully.")
        except sqlite3.Error as e:
            logger.error(f"Error setting up database: {e}")

    def record_activations_to_db(self, activations):
        """Records a list of activations to the database.
        
        Args:
            activations (list of tuples): Each tuple contains (layer, neuron, activation).
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.executemany("INSERT INTO activations (layer, neuron, activation) VALUES (?, ?, ?)", activations)
                conn.commit()
                logger.info(f"Successfully recorded {len(activations)} activations to the database.")
        except sqlite3.Error as e:
            logger.error(f"Error recording activations to the database: {e}")
