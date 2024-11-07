import logging
import re
import asyncio
import torch
from pathlib import Path
from datasets import Dataset
from telethon import TelegramClient
from telethon.errors import FloodWaitError
from telethon.tl.functions.messages import GetHistoryRequest
from transformers import pipeline
from message_scraping.scraping import parse_history


class SnowballSearch:
    """
    A class to perform snowball search for identifying potential pump channels on Telegram.

    This class iterates through known and discovered Telegram channels, analyzing messages to classify 
    channels based on pump-and-dump activity indicators. It leverages OpenAI's language model for classification 
    and Telethon for message scraping.

    Attributes:
        client (TelegramClient): A Telethon client instance for interacting with Telegram.
        max_snowball_depth (int): The maximum depth of recursive search for new channels.
        threshold_pump_channel (float): The threshold proportion of pump messages needed to classify a channel as a pump channel.
        known_pump_channels_dir (str): Path to the file containing known pump channels.
        new_channels_dir (str): Path to the directory where newly discovered pump channels are saved.

    Methods:
        extract_channel_links(message_text):
            Extracts potential Telegram channel links from a message.

        determine_if_pump_channel(messages):
            Asynchronously determines if a channel is likely a pump channel based on message classifications.

        load_known_pump_channels():
            Loads a list of known pump channels from a specified file.

        fetch_channel_messages(channel_name, total_count_limit=100):
            Asynchronously fetches messages from a specified Telegram channel up to a specified limit.
    """
    def __init__(
        self,
        client: TelegramClient,
        max_snowball_depth: int,
        threshold_pump_channel: float,
        known_pump_channels_dir: str,
        new_channels_dir: str
    ):
        self.client = client
        self.max_snowball_depth = max_snowball_depth
        self.threshold_pump_channel = threshold_pump_channel
        self.known_pump_channels_dir = known_pump_channels_dir
        self.new_channels_dir = new_channels_dir
        self.classifier = pipeline(
            'text-classification',
            model='./models/pump_bertweet',
            tokenizer='vinai/bertweet-base',
            device=0 if torch.cuda.is_available() else None,
            truncation=True, max_length=128
        )


    def extract_channel_links(self, message_text: str) -> list[str]:
        """
        Extracts Telegram channel links from a message.

        Args:
            message_text (str): The text of the message to parse.

        Returns:
            list[str]: A list of extracted channel usernames or IDs.
        """
        patterns = [
            r"(?:https?://)?t\.me/([\w_]+)",           # t.me/channel_name
            r"(?:https?://)?telegram\.me/([\w_]+)",    # telegram.me/channel_name
            r"@([\w_]+)",                              # @channel_name
        ]
        channel_links = []
        for pattern in patterns:
            matches = re.findall(pattern, message_text)
            channel_links.extend(matches)
        return channel_links


    async def determine_if_pump_channel(self, messages: list[str]):
        """
        Determines if a Telegram channel is likely to be a pump channel based on message classifications.

        Args:
            messages (list[str]): A list of message texts from the channel to classify.

        Returns:
            bool: True if the proportion of non-garbage classified messages (i.e., excluding class 5) 
                meets or exceeds the threshold for being a pump channel; otherwise, False.

        Raises:
            Exception: If an issue occurs during message classification (e.g., rate limit issues or device errors).
        """
        dataset = Dataset.from_dict({"text": messages})
        predictions = self.classifier(dataset["text"], batch_size=16, truncation=True, max_length=128)
        predicted_labels = [int(pred['label'].replace('LABEL_', '')) for pred in predictions]
        pump_proportion = (len(predicted_labels) - predicted_labels.count(5)) / len(predicted_labels)
        logging.info(f"Proportion of pump messages is {pump_proportion}")
        return pump_proportion > self.threshold_pump_channel


    def load_known_pump_channels(self):
        """
        Loads a list of known pump channels from a file.

        Returns:
            list[str]: A list of known pump channel identifiers.

        Logs:
            Error if the file containing known pump channels does not exist.
        """
        if not Path(self.known_pump_channels_dir).exists():
            logging.error(f"Known pump channels file {self.known_pump_channels_dir} does not exist.")
            return []
        with open(self.known_pump_channels_dir, "r") as f:
            channels = [line.strip() for line in f.readlines() if line.strip()]
        return channels


    async def fetch_channel_messages(self, channel_name: str, total_count_limit: int = 10_000):
        """
        Fetches messages from a specified Telegram channel up to a specified limit.

        Args:
            channel_name (str): The name or ID of the Telegram channel.
            total_count_limit (int, optional): The maximum number of messages to fetch. Defaults to 10,000.

        Returns:
            list[str]: A list of message contents fetched from the channel.

        Logs:
            Warning if a rate limit (FloodWaitError) is hit, along with the wait time in seconds.
            Error if there is an issue fetching messages from the specified channel.
        """
        all_messages = []
        offset_id = 0
        total_messages = 0
        batch_size = 100

        while True:
            try:
                history = await self.client(GetHistoryRequest(
                    peer=channel_name,
                    offset_id=offset_id,
                    offset_date=None,
                    add_offset=0,
                    limit=batch_size,
                    max_id=0,
                    min_id=0,
                    hash=0
                ))
            except FloodWaitError as e:
                logging.warning(f"Flood wait error. Waiting for {e.seconds} seconds.")
                await asyncio.sleep(e.seconds)
                continue
            except Exception as e:
                logging.error(f"Error fetching messages from {getattr(channel_name, 'username', 'unknown')} : {e}")
                break

            parsed_history = await parse_history(self.client, history)

            if not parsed_history:
                break

            for msg in parsed_history:
                if msg["message"]:
                    all_messages.append(msg["message"])

            offset_id = parsed_history[-1]['id']
            total_messages += len(parsed_history)

            if total_count_limit != 0 and total_messages >= total_count_limit:
                break

        return all_messages
