# Quick Start Guide

## Setup in 5 Minutes

### 1. Get Your Telegram ID
Talk to [@userinfobot](https://t.me/userinfobot) and copy your numeric ID.

### 2. Configure .env File
Open `.env` and add your ID:

```env
ALLOWED_USERS=your_id_here
```

**Example:**
```env
ALLOWED_USERS=123456789
```

For multiple users:
```env
ALLOWED_USERS=123456789,987654321,555666777
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Bot

```bash
python bot.py
```

You should see:
```
🚀 Bot starting...
🔒 Security mode: 1 authorized user(s)
✅ Bot ready - Polling mode activated
✅ Bot ready and listening for messages...
```

### 5. Test on Telegram

1. Find your bot on Telegram
2. Send `/start`
3. Try a conversation: "Hello, how are you?"
4. Try image generation: "Generate an image of a space cat"

---

## Quick Troubleshooting

### "You don't have permission to use this bot"
→ Your ID is not in `ALLOWED_USERS`. Verify you added the correct ID.

### "Missing required environment variables"
→ Verify `.env` has:
- `TELEGRAM_BOT_TOKEN`
- `AZURE_API_KEY`
- `AZURE_ENDPOINT`

### Bot doesn't respond
→ Verify the bot is running and check console logs.

---

## Full Documentation

- **[README.md](README.md)** - Complete documentation
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Cloud deployment guide

---

## Pre-Production Checklist

Before deploying to cloud:

- [ ] `ALLOWED_USERS` configured with valid IDs
- [ ] Tested locally successfully
- [ ] Environment variables verified
- [ ] Read [DEPLOYMENT.md](DEPLOYMENT.md) guide

---

Ready! Your bot is modernized and ready to use. 🎉
