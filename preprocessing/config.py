import logging
import re
from pathlib import Path

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Define global paths
RESSOURCE_DIR = Path('./data/internal/resources')


def load_data(filepath):
    with open(filepath,'r') as f:
        return {v.replace('\n', '').strip() for v in f.readlines()}


# Define suffixes to remove
suffixes = {
    '.com', '.io', '.pro', '.co', '.xyz', '.net', '.org', '.crypto', '.us', '.uk', '.ru', '.ch', '.org'
}

# Load exchange names
cex = load_data(RESSOURCE_DIR / 'cex.txt')
dex = load_data(RESSOURCE_DIR / 'dex.txt')
EXCHANGES = cex.union(dex)

# Pattern: exchange name can be preceded by whitespace, colon, or be at the beginning of a string
# Pattern: exchange name can be followed by whitespace, any sentence punctuation, or be at the end of a string
cex_pattern = re.compile(
    '|'.join(f"(?:^|(?<=\\s)){re.escape(exchange)}\\b" for exchange in sorted(cex, key=len, reverse=True)), re.IGNORECASE
)
dex_pattern = re.compile(
    '|'.join(f"(?:^|(?<=\\s)){re.escape(exchange)}\\b" for exchange in sorted(dex, key=len, reverse=True)), re.IGNORECASE
)
suffix_pattern = re.compile(
    r'(' + '|'.join(map(re.escape, suffixes)) + r')\b', re.IGNORECASE
)

# Create keyword config
REGEX_PATTERN = {
    'cex_pattern': cex_pattern,
    'dex_pattern': dex_pattern,
    'suffix_pattern': suffix_pattern
}
