import re
import tldextract


class TelegramMessageNormalizer:
    """
    A class to filter and preprocess Telegram messages based on given keywords.

    Attributes:
        unwanted_words (list): List of unwanted words.
        wanted_words (list): List of wanted words.
        exchanges (list): List of exchange names.
        symbols (list): List of token symbols.
        message_files (list): List of message file names to process.
    """

    def __init__(self, exchanges, regex_pattern, message_files=None, tokenizer=None):
        """
        Initializes the TelegramMessageFilter with keywords and message files.

        Args:
            exchanges (set): Set with all top 50 CEX + top 100 DEX.
            regex_pattern (dict): Dictionary containing various compiled regex patterns.
            message_files (list): List of message file paths to process.
        """
        self.exchanges = exchanges

        self.cex_pattern = regex_pattern['cex_pattern']
        self.dex_pattern = regex_pattern['dex_pattern']
        self.suffix_pattern = regex_pattern['suffix_pattern']

        self.message_files = message_files
        self.tokenizer = tokenizer

    def extract_trading_pair(self, url):
        # Regular expression to match the trading pair
        if any(e in url.lower() for e in ['xt.com', 'lbank', 'bitrue']):  # lbank. etc. have lower-case symbols in url
            match = re.search(r'/?([a-z0-9]+)[_]([a-z0-9]+)/?', url)
        else:
            match = re.search(r'/?([A-Z0-9]+)[-_/]([A-Z0-9]+)/?', url)
        if match:
            return f"{match.group(1)}-{match.group(2)}"
        return None

    def normalize_links(self, token):
        """
        Normalizes user tags and urls.

        Args:
            token (str): Token to parse.

        Returns:
            str: The normalized token.
        """
        if token.startswith('@'):
            return "@USER"
        elif re.match(r'^(http|Http|<http|www.)', token):
            token = token.replace('<', '').replace('>', '').replace('Http', 'http')
            domain = tldextract.extract(token).domain

            # Try if it's the link to a telegram group
            if domain == 't':
                return f"LINK TO TELEGRAM GROUP: {token.split('/')[-1]}"

            # Try if url is link to an exchange
            elif domain.lower() in self.exchanges:
                # Try to find trading pair in url
                trading_pair = self.extract_trading_pair(token)
                if trading_pair:
                    return f"EXCHANGE: {domain}; TRADING PAIR: {trading_pair}."
                else:
                    return domain

            # Ultimately replace with UNKNOWN_URL
            return f"UNKNOWN_URL: {domain}"
        else:
            return token

    def replace_emoji_by_text(self, message):
        emoji2text = {
            '0️⃣': '0',
            '1️⃣': '1',
            '2️⃣': '2',
            '3️⃣': '3',
            '4️⃣': '4',
            '5️⃣': '5',
            '6️⃣': '6',
            '7️⃣': '7',
            '8️⃣': '8',
            '9️⃣': '9',
            '🔟': '10',
            '#️⃣': '#',
            '🆗': 'OK',
            '🆙': 'UP!',
            '🆒': 'COOL',
            '🆕': 'NEW',
            '🆓': 'FREE',
            '❗': '!',
            '❕': '!',
            '❓': '?',
            '❔': '?',
            '0\u20e3': '0',  # Additional cases for when the variation selector is missing (\u20e3 only)
            '1\u20e3': '1',
            '2\u20e3': '2',
            '3\u20e3': '3',
            '4\u20e3': '4',
            '5\u20e3': '5',
            '6\u20e3': '6',
            '7\u20e3': '7',
            '8\u20e3': '8',
            '9\u20e3': '9',
        }
        pattern_a = re.compile(r'(\d️⃣|\d{1,2}️⃣)\s+')
        pattern_b = re.compile('|'.join(re.escape(emoji) for emoji in emoji2text.keys()))
        message = pattern_a.sub(r'\1', message)
        message = pattern_b.sub(lambda x: emoji2text[x.group()], message)
        return message

    def add_space_to_time_units(self, message):
        # Regular expression pattern to match number followed by time units
        pattern = re.compile(r'(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|days?)', re.IGNORECASE)
        # Substitute the pattern with number followed by a space and then the time unit
        return pattern.sub(r'\1 \2', message)

    def normalize_message(self, message):
        """
        Cleans the message by removing URLs, emojis, and normalizing whitespace.

        Args:
            message (str): The message to be cleaned.

        Returns:
            str: The cleaned message.
        """
        # Ensure message is properly encoded and decoded
        message = message.encode('utf-16', 'surrogatepass').decode('utf-16')

        # Replace newline and strip leading and trailing whitespaces
        message = (
            message
            .replace('\n', ' ')
            .replace('@ ', ' ')
            .replace('#', ' ')
            .replace('’', "'")
        )

        # Replace number emojis by real numbers
        message = self.replace_emoji_by_text(message)

        # Replace e.g. 4minutes by 4 minutes
        message = self.add_space_to_time_units(message)

        # Create tokens
        tokens = self.tokenizer.tokenize(message)

        # If there is only one token it is most likely a token symbol
        if len(tokens) == 1:
            message = self.normalize_links(tokens[0])
        else:
            message = ' '.join(self.normalize_links(token) for token in tokens)

        # Replace common suffixes such as .com
        message = re.sub(self.suffix_pattern, '', message)

        # Normalize whitespace
        message = re.sub(r'\s+', ' ', message)

        return message.strip()

    def augment_message(self, message):
        """
        Augments the message by adding additional information on identified exchanges and token symbols.

        Args:
            message (str): The input message to search and augment.

        Returns:
            str: The message with fuzzy matches augmented with their corresponding keyword.
        """
        # Define replacements
        replacements = {
            '_CEX': self.cex_pattern,
            '_DEX': self.dex_pattern,
        }

        for suffix, pattern in replacements.items():
            # To store the replacements as (start_index, end_index, replacement_string)
            replacements_list = []
            for match in re.finditer(pattern, message):
                start, end = match.span()
                replacements_list.append((start, end, f'{match.group(0)}{suffix}'))

            # Sort replacements by starting index in reverse order to avoid messing up the indices
            replacements_list.sort(reverse=True)

            # Perform the replacements
            for start, end, replacement in replacements_list:
                replacement = replacement.replace(' ', '_')
                message = message[:start] + replacement + message[end:]

        return message

    def pipeline(self, message):
        """
        Processes a message by cleaning and augmenting it.

        Args:
            message (str): The message to be processed.

        Returns:
            str: The processed message.
        """
        message = self.normalize_message(message)
        message = self.augment_message(message)
        return message
