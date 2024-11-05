with open("./config/proxy_list.txt", "r") as f:
    proxy_list = f.readlines()


def create_proxy_dict(proxy_str):
    parts = proxy_str.split(':')
    ip = parts[0]
    port = parts[1]
    username = parts[2]
    password = parts[3]
    proxy_url = f"http://{username}:{password}@{ip}:{port}"
    return {"https": proxy_url}


# Creating a list of proxy dictionaries
PROXIES = [create_proxy_dict(proxy) for proxy in proxy_list]
