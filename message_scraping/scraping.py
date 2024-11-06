import json
import logging
from pathlib import Path
import telethon.tl.types as types
from telethon import TelegramClient

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Define global paths
DATA_DIR = Path('./data/internal')
RAW_DIR = DATA_DIR / 'raw'
MEDIA_DIR = RAW_DIR / 'media'
PUMP_CHANNELS_FILE = DATA_DIR / 'resources' / 'seed_channels.txt'


async def structure_message(
    client: TelegramClient,
    msg: types.Message,
    store_media: bool = False
) -> dict:
    """
    Restructures relevant information from the message and downloads images to a dedicated folder.

    Args:
        client (TelegramClient): The telegram client to use for scraping.
        msg (types.Message): The Telegram message object.
        store_media (bool, optional): Wether or not to download media attached to a 
            message object, defaults to False.

    Returns:
        dict: A dictionary containing structured message information.
    """
    file_name = None
    if store_media:
        if isinstance(msg.media, types.MessageMediaPhoto):
            file_name = f'{msg.peer_id.channel_id}_message_{msg.id}.jpg'
            path = MEDIA_DIR / file_name
            await client.download_media(msg.media, str(path), thumb=-1)
    return {'id': msg.id,
            'date': msg.date.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'channel_id': msg.peer_id.channel_id,
            'message': msg.message,
            'views': msg.views,
            'image': file_name}


async def parse_history(client: TelegramClient, history: types.messages.ChannelMessages) -> list[dict]:
    """
    Parses message history into a simpler format and only keeps important fields.

    Args:
        client (TelegramClient): The telegram client to use for scraping.
        history (types.messages.ChannelMessages): The Telegram channel messages object.

    Returns:
        list[dict]: A list of dictionaries containing structured message information.
    """
    return [await structure_message(client, msg) for msg in history.messages if (msg.message or msg.media)]


def save_history(history: dict, channel_id: int):
    """
    Saves message history batch-wise to a JSONL file.

    Args:
        history (dict): The message history data to be saved.
        channel_id (int): The ID of the Telegram channel.
    """
    file_path = RAW_DIR / f'{channel_id}.jsonl'
    with open(file_path, "a") as f:
        for data in history:
            f.write(json.dumps(data) + "\n")


def sort_jsonl_by_id(file_path: str):
    """
    Reads stored message history and sorts it from newest to oldest by message ID.

    Args:
        file_path (str): The path to the JSONL file containing message history.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    data = [json.loads(line) for line in lines]
    # Sort the data by 'id' field
    sorted_data = sorted(data, key=lambda x: x['id'], reverse=True)
    with open(file_path, 'w') as f:
        for entry in sorted_data:
            f.write(json.dumps(entry) + '\n')
