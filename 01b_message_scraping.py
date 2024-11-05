import json
import logging
from pathlib import Path
import pandas as pd
import telethon.tl.types as types
from telethon import TelegramClient
from telethon.tl.functions.messages import GetHistoryRequest
from utils.telegram import API_ID, API_KEY

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

# Define global paths
DATA_DIR = Path('./data')
RAW_DIR = DATA_DIR / 'raw'
MEDIA_DIR = RAW_DIR / 'media'
PUMP_CHANNELS_FILE = DATA_DIR / 'resources' / 'pump_channels.txt'

# initialize telegram client with or without proxies
client = TelegramClient('anon', API_ID, API_KEY)


async def structure_message(msg: types.Message, store_media: bool = False) -> dict:
    """
    Restructures relevant information from the message and downloads images to a dedicated folder.

    Args:
        msg (types.Message): The Telegram message object.

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


async def parse_history(history: types.messages.ChannelMessages) -> list[dict]:
    """
    Parses message history into a simpler format and only keeps important fields.

    Args:
        history (types.messages.ChannelMessages): The Telegram channel messages object.

    Returns:
        list[dict]: A list of dictionaries containing structured message information.
    """
    return [await structure_message(msg) for msg in history.messages if (msg.message or msg.media)]


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


async def fetch_channel_history(channel_name: str, channel_id: int, min_id: int, total_count_limit: int = 500_000):
    """
    Fetches the history of a Telegram channel and saves the messages.

    Args:
        channel_name (str): The name of the Telegram channel.
        channel_id (int): The ID of the Telegram channel.
        min_id (int): The minimum message ID to start fetching from.
        total_count_limit (int, optional): The maximum number of messages to retrieve. Defaults to 50.
    """
    offset_id = 0
    total_messages = 0

    while True:
        logging.info(f"Current Offset ID: {offset_id}; Total Messages: {total_messages}")
        batch_history = await client(GetHistoryRequest(
            peer=channel_name,
            offset_id=offset_id,
            offset_date=None,
            add_offset=0,
            limit=500,
            max_id=0,
            min_id=min_id,
            hash=0
        ))

        parsed_history = await parse_history(batch_history)

        # Check if end is reached
        if len(parsed_history) == 0:
            break

        offset_id = parsed_history[-1]['id']
        total_messages += len(parsed_history)
        save_history(parsed_history, channel_id)

        if total_count_limit != 0 and total_messages >= total_count_limit:
            break

    sort_jsonl_by_id(RAW_DIR / f"{channel_id}.jsonl")


async def main():
    """
    Reads a list of pump channels and extracts the full history of the channel if it does not already exist,
    otherwise adds only the newest messages.
    """
    if not PUMP_CHANNELS_FILE.exists():
        logging.error(f"Pump channels file {PUMP_CHANNELS_FILE} does not exist.")
        return

    with open(PUMP_CHANNELS_FILE, "r") as f:
        channels = f.readlines()

    existing_channels = {file.stem for file in RAW_DIR.glob("*.jsonl")}

    for channel_url in channels:
        channel_name = channel_url.strip().split('/')[-1]

        try:
            channel = await client.get_entity(channel_name)
            if not channel.broadcast:
                logging.warning(f"Channel: {channel_name} is not a broadcast channel")
                continue
        except Exception as e:
            logging.error(f"Channel: {channel_name} cannot be found! Error: {e}")
            continue

        min_id = 0
        if str(channel.id) in existing_channels:
            logging.info(f"History for channel {channel_name} already exists, fetching only new messages!")
            min_id = pd.read_json(RAW_DIR / f"{channel.id}.jsonl", lines=True, nrows=1).id.iloc[0]

        await fetch_channel_history(channel_name, channel.id, min_id)


if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main())
