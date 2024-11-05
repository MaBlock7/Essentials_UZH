import logging
from pathlib import Path
import pandas as pd
from preprocessing.standard_normalizer import StandardNormalizer

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Define global paths
BASE_DIR = Path('./data/internal')
SOURCE_DIR = BASE_DIR / 'raw'
TARGET_DIR = BASE_DIR / 'processed'


def create_message_files_set(directory):
    return {file for file in directory.iterdir() if str(file).endswith('.jsonl')}


def main(normalizer):

    # Create list of paths to all message history files
    message_files = create_message_files_set(SOURCE_DIR)
    logging.info(f'Found {len(message_files)} files with messages.')

    for file_path in message_files:
        logging.info(f'Processing file: {file_path}')
        df = pd.read_json(file_path, lines=True)
        df = df[df.message != ""].copy()
        df = df.dropna(subset=['message'])
        df['message'] = df['message'].apply(normalizer.pipeline)
        output_path = TARGET_DIR / file_path.name.replace('.jsonl', '.csv')
        df.to_csv(output_path, index=False)
        logging.info(f'Saved processed file to: {output_path}')


if __name__ == '__main__':
    main(StandardNormalizer)
