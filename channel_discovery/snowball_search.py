import logging
import re
import asyncio
import torch
from pathlib import Path
from collections import deque
from telethon.errors import FloodWaitError
from telethon.tl.functions.messages import GetHistoryRequest

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding

from train_bertweet import MessagesDataset
import message_scraping


class SnowballSearch():
    """
    A class to perform snowball search for identifying potential pump channels on Telegram.

    This class iterates through known and discovered Telegram channels, analyzing messages to classify 
    channels based on pump-and-dump activity indicators. It leverages OpenAI's language model for classification 
    and Telethon for message scraping.

    Attributes:
        max_snowball_depth (int): The maximum depth of recursive search for new channels.
        threshold_pump_channel (float): The threshold proportion of pump messages needed to classify a channel as a pump channel.
        known_pump_channels_dir (str): Path to the file containing known pump channels.
        new_channels_dir (str): Path to the directory where newly discovered pump channels are saved.
        client (TelegramClient): A Telethon client instance for interacting with Telegram.

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
    def __init__(self, max_snowball_depth, threshold_pump_channel, known_pump_channels_dir, new_channels_dir):
        self.client = message_scraping.client
        self.max_snowball_depth = max_snowball_depth
        self.threshold_pump_channel = threshold_pump_channel
        self.known_pump_channels_dir = known_pump_channels_dir
        self.new_channels_dir = new_channels_dir

        self.tokenizer = AutoTokenizer.from_pretrained('vinai/bertweet-base', use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            './models/bertweet_final',
            num_labels=6
        )
        self.model.eval()


    def extract_channel_links(self, message_text):
        """
        Extracts Telegram channel links from a message.

        Args:
            message_text (str): The text of the message to parse.

        Returns:
            list[str]: A list of extracted channel usernames or IDs.
        """
        channel_links = []

        patterns = [
            r"(?:https?://)?t\.me/([\w_]+)",           # t.me/channel_name
            r"(?:https?://)?telegram\.me/([\w_]+)",    # telegram.me/channel_name
            r"@([\w_]+)",                              # @channel_name
        ]

        for pattern in patterns:
            matches = re.findall(pattern, message_text)
            channel_links.extend(matches)

        return channel_links


    async def determine_if_pump_channel(self, messages):
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
        dummy_labels = [0] * len(messages)
        dataset = MessagesDataset(texts=messages, labels=dummy_labels, tokenizer=self.tokenizer, max_len=128)
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=16, collate_fn=data_collator)

        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = {k: v.to(self.model.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend(batch_predictions)

        pump_proportion = (len(predictions) - predictions.count(5)) / len(predictions)
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


    async def fetch_channel_messages(self, channel_name, total_count_limit: int = 100): ### INCREASE total_count_limit!
        """
        Fetches messages from a specified Telegram channel up to a specified limit.

        Args:
            channel_name (str): The name or ID of the Telegram channel.
            total_count_limit (int, optional): The maximum number of messages to fetch. Defaults to 100.

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

            parsed_history = await message_scraping.parse_history(history)

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


async def main(S):
    known_pump_channels = S.load_known_pump_channels()

    visited_channels = set()
    pump_channels = set(known_pump_channels)
    queue = deque()
    max_depth = S.max_snowball_depth

    for channel in known_pump_channels:
        queue.append((channel, 0))

    while queue:
        current_channel, depth = queue.popleft()

        if depth > max_depth:
            continue

        if current_channel in visited_channels:
            continue

        visited_channels.add(current_channel)

        logging.info(f"Processing channel: {current_channel} at depth: {depth}")

        try:
            channel_entity = await S.client.get_entity(current_channel)
        except Exception as e:
            logging.error(f"Failed to get entity for channel {current_channel}: {e}")
            continue

        # Fetch messages from the channel
        messages = await S.fetch_channel_messages(channel_entity)

        if not messages:
            logging.warning(f"No messages found in channel {current_channel}")
            continue

        # Classify the channel and add it to pump_channels if classified as pump channel
        if current_channel not in known_pump_channels:
            is_pump_channel = await S.determine_if_pump_channel(messages)
            if is_pump_channel:
                logging.info(f"{"\u2705"} Channel {current_channel} classified as pump channel.")
                pump_channels.add(current_channel)
                with open(Path(S.new_channels_dir) / 'discovered_pump_channels.txt', 'a') as f:
                    f.write(f"https://t.me/{current_channel}\n")
            else:
                logging.info(f"{"\u274C"} Channel {current_channel} is not a pump channel.")

        # Extract channel links from messages
        new_channels = set()
        for msg in messages:
            links = S.extract_channel_links(msg)
            new_channels.update(links)

        # Add new channels to the queue
        for new_channel in new_channels:
            if new_channel not in visited_channels:
                queue.append((new_channel, depth + 1))

    logging.info(f"Found {len(pump_channels)} pump channels.")


if __name__ == '__main__':
    S = SnowballSearch(
        max_snowball_depth=2,
        threshold_pump_channel=0.3,
        known_pump_channels_dir='data/internal/resources/pump_channels.txt',
        new_channels_dir='data/internal/resources/discovered_channels'
    )

    with S.client:
        S.client.loop.run_until_complete(main(S=S))
