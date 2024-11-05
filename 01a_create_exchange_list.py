import requests
from requests.exceptions import RequestException, Timeout
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils.proxies import PROXIES

# Define global paths
DATA_DIR = Path('./data/resources')


def test_and_filter_proxies(proxies, test_url='https://www.google.com', timeout=1.5):
    """
    Tests a list of proxy servers and removes the ones that fail to connect.

    Args:
    proxies (list): A list of proxy server addresses in the format 'http://<ip>:<port>'.
    test_url (str): The URL to test the proxy servers with.
    timeout (int): The maximum time in seconds to wait for a response from the proxy server.

    Returns:
    list: A list of working proxy servers.
    """
    working_proxies = []
    for proxy in tqdm(proxies):
        try:
            response = requests.get(test_url, proxies=proxy, timeout=timeout)
            response.raise_for_status()  # Raise an exception for HTTP errors
            working_proxies.append(proxy)
        except (RequestException, Timeout):
            continue
    return working_proxies


def fetch_data_with_proxies(proxies, url, timeout=1.5):
    """
    Fetches data from a given URL using a list of proxy servers.

    Args:
    proxies (list): A list of proxy server addresses in the format 'http://<ip>:<port>'.
    url (str): The URL to fetch data from.
    timeout (int): The maximum time in seconds to wait for a response from the proxy server.

    Returns:
    dict: The JSON response from the server.
    """
    for proxy in proxies:
        try:
            response = requests.get(url, proxies=proxy, timeout=timeout)
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except (RequestException, Timeout):
            continue
    raise Exception("All proxies failed")


if __name__ == '__main__':

    working_proxies = test_and_filter_proxies(PROXIES)
    print(f'Number of working proxies: {len(working_proxies)}')

    if not working_proxies:
        raise Exception("No working proxies available")

    try:
        data = fetch_data_with_proxies(working_proxies, 'https://api.coingecko.com/api/v3//exchanges/list')
        df = pd.DataFrame(data)
        df['centralized'] = 0
        df['trade_volume_24h_btc_normalized'] = 0.0
    except Exception as e:
        print(str(e))
    print(f'Number of exchanges found: {len(df)}')

    for id in tqdm(df['id']):
        try:
            data = fetch_data_with_proxies(working_proxies, f'https://api.coingecko.com/api/v3//exchanges/{id}')
            df.loc[df['id'] == id, 'centralized'] = 1 if data['centralized'] else 0
            df.loc[df['id'] == id, 'trade_volume_24h_btc_normalized'] = data['trade_volume_24h_btc_normalized']
        except Exception as e:
            print(str(e))

    df.to_csv(DATA_DIR / 'exchanges.csv', index=False)
