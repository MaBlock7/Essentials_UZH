import shutil
from os import getenv
from os import path as path
from typing import Any

import dotenv


def set_env():
    """
    Check if .env file exists, if it does, load it to the environment, otherwise create new .env file
    """
    # Check if .env file exists
    dotenv_path = path.abspath(path.join(__file__, "../auth/.env"))
    found = path.exists(dotenv_path)

    # Create .env file from template if .env not found
    if not found:
        print(".env file not found\nCreating .env file")
        shutil.copy(f"{dotenv_path}.template", f"{dotenv_path}")
        raise SystemExit(
            f"Please complete the newly created .env file by inserting your credentials\n> {dotenv_path}"
        )

    # Load variables into environment
    dotenv.load_dotenv(dotenv_path)


def get_env_values(value: str, alternative: Any = None) -> Any:
    """
    Retrieve values from environment.
    """
    # Check if code is running on server (environment variable ON_SERVER is set). If so, directly read the environment
    # variables. If not, look for local .env file and load variables to the environment before reading them.
    if not getenv("ON_SERVER"):
        set_env()
    variable = getenv(value)
    if variable == '':
        return alternative
    else:
        return variable
