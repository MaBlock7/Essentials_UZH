import logging
import pandas as pd
from telethon.tl.functions.messages import GetHistoryRequest
from message_scraping.scraping import *
from utils.telegram import API_ID, API_KEY


async def fetch_channel_history(
    client: TelegramClient,
    channel_name: str,
    channel_id: int,
    min_id: int,
    total_count_limit: int = 500_000
):
    """
    Fetches the history of a Telegram channel and saves the messages.

    Args:
        client (TelegramClient): The telegram client to use for scraping.
        channel_name (str): The name of the Telegram channel.
        channel_id (int): The ID of the Telegram channel.
        min_id (int): The minimum message ID to start fetching from.
        total_count_limit (int, optional): The maximum number of messages to retrieve. Defaults to 500k.
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

        parsed_history = await parse_history(client, batch_history)

        # Check if end is reached
        if len(parsed_history) == 0:
            break

        offset_id = parsed_history[-1]['id']
        total_messages += len(parsed_history)
        save_history(parsed_history, channel_id)

        if total_count_limit != 0 and total_messages >= total_count_limit:
            break

    sort_jsonl_by_id(RAW_DIR / f"{channel_id}.jsonl")


async def main(client: TelegramClient):
    """
    Reads a list of pump channels and extracts the full history of the channel if 
    it does not already exist, otherwise adds only the newest messages.
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

        await fetch_channel_history(client, channel_name, channel.id, min_id)


if __name__ == '__main__':

    client = TelegramClient('anon', API_ID, API_KEY)

    with client:
        client.loop.run_until_complete(main(client))
