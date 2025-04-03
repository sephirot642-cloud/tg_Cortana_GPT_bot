# Telegram Bot with ChatGPT and DALL-E

This Telegram bot integrates ChatGPT and DALL-E capabilities to provide AI-generated text and image responses.

## Features

- 💬 **Chat with memory**: Maintains conversation context for more coherent responses
- 🖼️ **Image generation**: Creates high-definition images with DALL-E 3
- 🔄 **Multiple models**: Support for various OpenAI models (GPT-4, GPT-3.5, etc.)
- 📊 **Usage control**: Tracks token usage per user
- 🔒 **Configurable limits**: Set usage limits to control costs
- 📝 **Detailed logging**: Complete activity logs for monitoring
- 🔁 **Automatic retries**: Error handling with retries for increased robustness

## Requirements

- Python 3.7 or higher
- An account with Azure OpenAI or OpenAI
- A Telegram bot token (obtained via [@BotFather](https://t.me/BotFather))

## Installation

### Local Installation

1. Clone this repository or download the files

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure the `.env` file with your credentials:

```
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
TELEGRAM_BOT_TOKEN=your_telegram_token_here

# Optional configuration
MAX_TOKENS_PER_USER=1000
MAX_CONVERSATIONS_STORED=10
LOG_LEVEL=INFO
```

### Deployment on Render.com

1. Create an account on [Render.com](https://render.com) if you don't have one yet

2. Click on "New" and select "Web Service"

3. Connect your GitHub repository or upload the files manually

4. Configure the service with the following parameters:
   - **Name**: Your bot's name
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn bot:app`

5. In the "Environment Variables" section, add the following variables:

```
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=your_endpoint_here
TELEGRAM_BOT_TOKEN=your_telegram_token_here
RENDER=true
WEBHOOK_URL=https://your-app-name.onrender.com
```

6. Click on "Create Web Service"

7. Once deployed, set up the Telegram webhook by visiting:
   `https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook?url=https://your-app-name.onrender.com/<TELEGRAM_BOT_TOKEN>`

## Usage

1. Start the bot:

```bash
python bot.py
```

2. Open Telegram and search for your bot by its username

3. Start a conversation with the `/start` command

## Available Commands

- `/start` - Starts the bot and shows the welcome message
- `/help` - Shows a detailed help guide
- `/setmodel <name>` - Changes the AI model to use
- `/generatehd <description>` - Generates a high-definition image (1920x1080)
- `/reset` - Resets the current conversation, removing the context
- `/usage` - Shows your current usage statistics

## Customization

You can modify the following parameters in the `.env` file:

- `MAX_TOKENS_PER_USER`: Token limit per user
- `MAX_CONVERSATIONS_STORED`: Maximum number of stored conversations
- `LOG_LEVEL`: Log detail level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Troubleshooting

- If the bot doesn't respond, check that the credentials in the `.env` file are correct
- Check the logs in `bot.log` to identify possible errors
- Make sure you have enough credits in your OpenAI account
- For issues on Render.com, check the service logs in the Render dashboard
- If the webhook doesn't work, ensure the URL is publicly accessible

## License

This project is available under the MIT license.