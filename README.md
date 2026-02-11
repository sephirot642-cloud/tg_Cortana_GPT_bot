# Telegram Bot with GPT-4o and DALL-E 3

Modern Telegram bot with GPT-4o and DALL-E 3 integration, optimized for cloud deployment with async architecture and robust security measures.

## Features

- 🤖 **GPT-4o with Function Calling**: Intelligent image generation through tools
- 🎨 **DALL-E 3**: HD image creation based on descriptions
- 💬 **Conversation Memory**: Maintains context for coherent responses
- 🔒 **Access Control**: User whitelist for authorized access
- ⚡ **Async Architecture**: AsyncAzureOpenAI for better performance
- 📝 **Structured Logging**: Complete activity monitoring
- 🌐 **Cloud-Ready**: Ready for deployment on Render, Railway, etc.

## Requirements

- Python 3.10 or higher
- Azure OpenAI account with GPT-4o and DALL-E 3 access
- Telegram bot token (get one from [@BotFather](https://t.me/BotFather))
- Your Telegram user ID (get it from [@userinfobot](https://t.me/userinfobot))

## Installation

### Local Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository>
   cd tg_Cortana_GPT_bot
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Copy `.env.example` to `.env`:
   ```bash
   copy .env.example .env  # Windows
   # or
   cp .env.example .env  # Linux/Mac
   ```
   
   Edit `.env` with your credentials:
   ```env
   AZURE_API_KEY=your_azure_key
   AZURE_ENDPOINT=https://your-resource.openai.azure.com
   AZURE_API_VERSION=2024-05-01-preview
   TELEGRAM_BOT_TOKEN=your_telegram_token
   ALLOWED_USERS=123456789,987654321
   ```

5. **Run the bot**
   ```bash
   python bot.py
   ```

### Security Configuration

**⚠️ IMPORTANT:** For production use, you **MUST** configure `ALLOWED_USERS`:

1. Get your Telegram ID from [@userinfobot](https://t.me/userinfobot)
2. Add your ID (and other authorized users) to `ALLOWED_USERS` in `.env`
3. Example: `ALLOWED_USERS=123456789,987654321`

If `ALLOWED_USERS` is empty, **anyone** can use your bot (❌ **NOT RECOMMENDED** for production).

## Cloud Deployment

### Render.com

1. Create account on [Render.com](https://render.com)
2. Create new "Web Service"
3. Connect your GitHub repository
4. Configure environment variables in Render dashboard
5. Start command: `python bot.py`

### Railway.app

1. Create account on [Railway.app](https://railway.app)
2. Create new project from GitHub
3. Configure environment variables
4. Railway auto-detects `requirements.txt`

### Polling Mode (Any Server)

The bot runs in polling mode by default, which means **no webhook configuration needed**. Simply run:

```bash
python bot.py
```

Works on any server, VPS, or local machine.

## Usage

### Available Commands

- `/start` - Show welcome message
- `/clear` - Clear conversation memory

### Examples

**Normal conversation:**
```
User: What is artificial intelligence?
Bot: [Detailed GPT-4o response]
```

**Image generation (automatic via function calling):**
```
User: Generate an image of a cyberpunk cat in a futuristic city
Bot: 🎨 Generating image: A cyberpunk cat in a futuristic city...
Bot: [Sends generated image]
```

The bot automatically detects when to generate images thanks to GPT-4o Function Calling.

## Configuration

### Environment Variables

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token | ✅ | `123456:ABC-DEF...` |
| `AZURE_API_KEY` | Azure OpenAI API key | ✅ | `abc123...` |
| `AZURE_ENDPOINT` | Azure OpenAI endpoint | ✅ | `https://....openai.azure.com` |
| `AZURE_API_VERSION` | Azure OpenAI API version | ❌ | `2024-05-01-preview` |
| `ALLOWED_USERS` | Authorized user IDs | ⚠️ Recommended | `123,456,789` |
| `RENDER` | Production mode (Render.com) | ❌ | `true` or `false` |

### Azure Deployments

Ensure your Azure OpenAI deployments have these names:
- **GPT-4o**: `gpt-4o`
- **DALL-E 3**: `dall-e-3`

If your deployments have different names, edit the `AzureDeployment` enum in `bot.py`:

```python
class AzureDeployment(Enum):
    GPT4O = "your-gpt4o-deployment-name"
    DALLE3 = "your-dalle3-deployment-name"
```

## Troubleshooting

### Bot doesn't respond

- ✅ Verify credentials in `.env` are correct
- ✅ Ensure your ID is in `ALLOWED_USERS`
- ✅ Check console logs for errors
- ✅ Verify you have credits in Azure OpenAI account

### API version error

If you receive API version errors:
- Verify your Azure OpenAI supports `2024-05-01-preview`
- You can change the version in `bot.py` line 54

### Bot doesn't generate images

- ✅ Verify you have a DALL-E 3 deployment in Azure
- ✅ Ensure deployment name is `dall-e-3`
- ✅ Check logs for specific error

### Access denied

If you receive "⛔ You don't have permission to use this bot":
- Your Telegram ID is not in `ALLOWED_USERS`
- Get your ID from [@userinfobot](https://t.me/userinfobot)
- Add it to `ALLOWED_USERS` in `.env`

## Architecture

```
bot.py
├── AsyncAzureOpenAI Client (async API)
├── ConversationManager (history management)
├── @restricted decorator (access control)
├── Handlers
│   ├── /start (welcome)
│   ├── /clear (clear memory)
│   └── chat_handler (messages + function calling)
└── Tools Schema (DALL-E 3 integration)
```

## Security

- ✅ User whitelist (`ALLOWED_USERS`)
- ✅ Environment variable validation
- ✅ Logging of unauthorized access attempts
- ✅ No sensitive data stored on disk
- ⚠️ For production, consider adding rate limiting

## License

This project is available under the MIT license.

## Contributing

Contributions are welcome. Please:
1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request