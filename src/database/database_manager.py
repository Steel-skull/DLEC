import sqlite3
import logging
logger = logging.getLogger(__name__)

class DatabaseManager:

    def __init__(self, db_path):
        self.db_path = db_path
        
    def create_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
        except sqlite3.Error as e:
            logger.error(f'Error connecting to the database: {e}')
        return conn

    def setup_database(self):
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            cursor.execute('CREATE TABLE IF NOT EXISTS activations (\n                                  layer TEXT,\n                                  neuron INTEGER,\n                                  activation REAL)')
            conn.commit()
            conn.close()
            logger.info('Database setup successfully.')
        except sqlite3.Error as e:
            logger.error(f'Error setting up database: {e}')

    def record_activations_to_db(self, activations):
        try:
            conn = self.create_connection()
            cursor = conn.cursor()
            cursor.executemany('INSERT INTO activations (layer, neuron, activation) VALUES (?, ?, ?)', activations)
            conn.commit()
            conn.close()
            logger.info(f'Successfully recorded {len(activations)} activations to the database.')
        except sqlite3.Error as e:
            logger.error(f'Error recording activations to the database: {e}')
