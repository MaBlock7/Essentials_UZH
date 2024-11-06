import logging
from pathlib import Path
from collections import deque
from telethon import TelegramClient
from utils.telegram import API_ID, API_KEY
from channel_discovery import SnowballSearch


def load_searched_channels(dir: str = 'data/internal/resources/searched_channels.txt') -> list[str]:
    if Path(dir).exists():
        with open(dir, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    else:
        return []


async def main(search: SnowballSearch):

    known_pump_channels = search.load_known_pump_channels()
    searched_pump_channels = load_searched_channels()

    visited_channels = set()
    new_pump_channels = set(searched_pump_channels)
    queue = deque()
    max_depth = search.max_snowball_depth

    critical_error = False

    for seed_channel in [c for c in known_pump_channels if c not in searched_pump_channels]:

        queue.append((seed_channel, 0))

        while queue:
            current_channel, depth = queue.popleft()

            if current_channel in visited_channels:
                continue

            visited_channels.add(current_channel)

            logging.info(f"Processing channel: {current_channel} at depth: {depth}")

            try:
                channel_entity = await search.client.get_entity(current_channel)
            except Exception as e:
                error_message = str(e)
                if "A wait of" in error_message and "is required (caused by ResolveUsernameRequest)" in error_message:
                    logging.error(f"Critical error encountered: {error_message}. Stopping the search.")
                    critical_error = True
                    break
                else:
                    logging.error(f"Failed to get entity for channel {current_channel}: {e}")
                    continue

            # Fetch messages from the channel
            messages = await search.fetch_channel_messages(channel_entity, total_count_limit=5_000 if depth < max_depth else 100)

            if not messages:
                logging.warning(f"No messages found in channel {current_channel} ({channel_entity.id})")
                continue
            else:
                logging.info(f"Fetched {len(messages)} from {current_channel} ({channel_entity.id})")

            # Classify the channel and add it to pump_channels if classified as pump channel
            if current_channel not in known_pump_channels:
                # Determine the proportion of pump-related messages within the 100 most recent messages
                is_pump_channel = await search.determine_if_pump_channel(messages[:100])
                if is_pump_channel:
                    logging.info(f"{"\u2705"} Channel {current_channel} ({channel_entity.id}) classified as pump channel.")
                    new_pump_channels.add(current_channel)
                    with open(Path(search.new_channels_dir), 'a') as f:
                        f.write(f"https://t.me/{current_channel}\n")
                else:
                    logging.info(f"{"\u274C"} Channel {current_channel} is not a pump channel.")

            # If we have reached the max depth level, we stop searching for more channel links
            if depth >= max_depth:
                continue

            # Extract channel links from messages
            new_channels = set()
            for msg in messages:
                links = search.extract_channel_links(msg)
                new_channels.update(links)

            logging.info(f"Found {len(new_channels)} Telegram links in {current_channel}")

            # Add new channels to the queue
            for new_channel in new_channels:
                seen_channels = visited_channels | set(known_pump_channels)
                if new_channel not in seen_channels:
                    queue.appendleft((new_channel, depth + 1))

        if critical_error:
            break

        with open('data/internal/resources/searched_channels.txt', 'a') as f:
            f.write(f"{seed_channel}\n")

    logging.info(f"Found {len(new_pump_channels)} additional pump channels.")


if __name__ == '__main__':

    search = SnowballSearch(
        client=TelegramClient('anon', API_ID, API_KEY),
        max_snowball_depth=2,  # we search 2 levels down from our seed channels
        threshold_pump_channel=0.3,  # if more than 30% of messages are pump-related, we consider it as a pump channel
        known_pump_channels_dir='data/internal/resources/seed_channels.txt',
        new_channels_dir='data/internal/resources/discovered_channels.txt'
    )

    with search.client:
        search.client.loop.run_until_complete(main(search))
