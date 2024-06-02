# LOAD SECRETS

from dotenv import load_dotenv

try:
    load_dotenv("./secrets.toml", verbose=True)
except:
    print(
        "No secrets.toml file found."
        " Make sure appropriate environment variables are set."
        " (e.g. OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)"
    )
