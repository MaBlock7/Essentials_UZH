import asyncio
import logging
import re
from collections import deque
from pathlib import Path
import torch
from telethon import TelegramClient, events, functions
from telethon.errors import (
    ChannelPrivateError,
    FloodWaitError,
    InviteHashExpiredError,
    UserAlreadyParticipantError,
)
from transformers import pipeline
from preprocessing import StandardNormalizer
from utils.telegram import API_ID, API_KEY
from utils.utils import get_env_values

logging.basicConfig(level=logging.INFO)


class ExchangeWarner:
    """
    Monitors Telegram channels for cryptocurrency pump and dump messages and sends notifications.

    Attributes:
        phone_number (str): The phone number associated with the Telegram client.
        client (TelegramClient): The Telegram client used for interacting with channels.
        classifier (Pipeline): The NLP model used to classify messages.
        _cache (dict): A cache of channels and their recent exchanges.
        channels (list[str]): List of channels to monitor.
    """

    def __init__(self, api_id: int, api_key: str, phone_number: str, subscribe: bool = False) -> None:
        """
        Initializes the ExchangeWarner class with API credentials and phone number.

        Args:
            api_id (int): Telegram API ID.
            api_key (str): Telegram API hash.
            phone_number (str): Phone number for Telegram account.
        """
        self.phone_number = phone_number
        self.client = TelegramClient('anon', api_id, api_key)
        self.classifier = pipeline(
            'text-classification',
            model='./models/pump_bertweet',
            tokenizer='vinai/bertweet-base',
            device=0 if torch.cuda.is_available() else None,
            truncation=True, max_length=128
        )
        self._cache: dict[str, deque[str]] = {}  # channel, exchange cache
        self.channels = self._load_channels()
        self.subscribe = subscribe

    async def start(self) -> None:
        """
        Starts the Telegram client, subscribes to channels, and monitors them for messages.
        """
        await self.client.start(phone=self.phone_number)
        logging.info("Telegram client started.")
        if self.subscribe:
            await self.subscribe_to_channels()
        self.client.add_event_handler(
            self.monitor_channels,
            events.NewMessage(chats=self.channels, incoming=True, outgoing=True, forwards=False)
        )
        await self.client.run_until_disconnected()

    async def subscribe_to_channels(self) -> None:
        """
        Ensures the phone number is subscribed to all monitored channels.
        """
        for channel in self.channels:
            while True:
                try:
                    await self.client(functions.channels.JoinChannelRequest(channel))
                    logging.info(f"Successfully joined channel: {channel}")
                    break
                except UserAlreadyParticipantError:
                    logging.info(f"Already a participant in the channel: {channel}")
                    break
                except ChannelPrivateError:
                    logging.warning(f"Cannot join private channel: {channel}")
                    break
                except InviteHashExpiredError:
                    logging.warning(f"Invite link for channel expired: {channel}")
                    break
                except FloodWaitError as e:
                    wait_time = e.seconds
                    logging.error(f"Failed to join channel {channel}: Wait {wait_time} seconds")
                    await asyncio.sleep(wait_time)
                except Exception as e:
                    logging.error(f"Failed to join channel {channel}: {e}")
                    break

    async def monitor_channels(self, event: events.NewMessage.Event) -> None:
        """
        Processes new messages from monitored channels and sends notifications based on message content.

        Args:
            event (events.NewMessage.Event): The message event containing the message data.
        """
        channel_id = event.chat.id
        channel_name = event.chat.title
        message_date, message_raw = event.date, event.raw_text
        logging.info(f"New message received from channel {channel_name} ({channel_id})")

        message_norm = StandardNormalizer.pipeline(message_raw)
        label, confidence = self._predict_message(message_norm)

        if label == 0:
            exchange = self.extract_exchange_name(message_norm)
            self.enqueue_exchange(channel_id, exchange)
            notification_message = (
                f"NEW PUMP ANNOUNCEMENT FOUND!\n\n"
                f"Channel: {channel_name}\n"
                f"On: {exchange}\n"
                f"Confidence: {confidence}\n"
                f"Original message:\n\n"
                f"{message_raw}"
            )
            await self.send_notification(notification_message)
        elif label == 4:
            exchange = self.dequeue_exchange(channel_id)
            notification_message = (
                f"PUMP CANCELLED OR POSTPONED!\n\n"
                f"Channel: {channel_name}\n"
                f"On: {exchange}\n"
                f"Confidence: {confidence}\n"
                f"Original message:\n\n"
                f"{message_raw}"
            )
            await self.send_notification(notification_message)
        elif label == 2:
            exchange = self.dequeue_exchange(channel_id)
            notification_message = (
                f"TARGET COIN RELEASED!\n\n"
                f"Channel: {channel_name}\n"
                f"On: {exchange}\n"
                f"Confidence: {confidence}\n"
                f"Original message:\n\n"
                f"{message_raw}"
            )
            await self.send_notification(notification_message)

    async def send_notification(self, notification: str) -> None:
        """
        Sends a notification message to a specific Telegram channel.

        Args:
            notification (str): The augmented message.
        """
        target_channel_name = get_env_values('CHANNEL_USER_NAME')
        await self.client.send_message(target_channel_name, notification)

    @staticmethod
    def extract_exchange_name(message: str) -> str | None:
        """
        Extracts the exchange name preceding '_CEX' or '_DEX' from a message.

        Args:
            message (str): The message to search.

        Returns:
            str | None: The extracted exchange name or None if not found.
        """
        match = re.search(r'(\w+)(?=_(?:CEX|DEX))', message)
        return match.group(1) if match else None

    def _load_channels(self, path: str = 'data/internal/resources/pump_channels.txt') -> list[str]:
        """
        Loads a list of known pump channels from a file.

        Args:
            path (str): Path to the file containing channel identifiers.

        Returns:
            list[str]: A list of known pump channel identifiers.
        """
        if not Path(path).exists():
            logging.error(f"Known pump channels file {path} does not exist.")
            return []
        with open(path, 'r') as f:
            return ['@' + channel.strip().split('/')[-1] for channel in f.readlines()]

    def _predict_message(self, message: str) -> tuple[int, float]:
        """
        Classifies the message into six classes.

        Args:
            message (str): The normalized message from the pump channel.

        Returns:
            tuple[int, float]: The label and confidence score for the message.
        """
        if message:
            output = self.classifier(message)
            return int(output[0]['label']), output[0]['score']
        return 5, 1.0

    def enqueue_exchange(self, channel_id: str, exchange: str) -> None:
        """
        Enqueues an exchange for a given channel.

        Args:
            channel_id (str): The ID of the channel.
            exchange (str): The exchange name.
        """
        if channel_id not in self._cache:
            self._cache[channel_id] = deque()
        self._cache[channel_id].append(exchange)

    def dequeue_exchange(self, channel_id: str) -> str | None:
        """
        Removes and returns the oldest exchange for a given channel.

        Args:
            channel_id (str): The ID of the channel.

        Returns:
            str | None: The dequeued exchange, or None if no exchange is cached.
        """
        if channel_id in self._cache and self._cache[channel_id]:
            return self._cache[channel_id].popleft()
        return None

if __name__ == '__main__':
    warner = ExchangeWarner(API_ID, API_KEY, get_env_values('PHONE_NUMBER'))
    asyncio.run(warner.start())
