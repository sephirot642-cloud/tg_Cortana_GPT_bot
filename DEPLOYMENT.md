# Deployment Guide - Telegram Bot

## Deployment Options

### 1. Polling Mode (Recommended for Getting Started)

**Advantages:**
- ✅ Works on any server
- ✅ No public domain required
- ✅ Simple configuration
- ✅ Ideal for development and small-scale production

**Disadvantages:**
- ❌ Higher resource consumption (constant polling)
- ❌ Slightly higher latency

**How to use:**
```bash
python bot.py
```

---

### 2. Render.com (Recommended for Production)

**Step-by-step:**

1. **Create Render.com account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

2. **Create new Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Service configuration:**
   ```
   Name: telegram-bot-gpt4o
   Environment: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: python bot.py
   ```

4. **Environment variables:**
   ```
   TELEGRAM_BOT_TOKEN=your_token_here
   AZURE_API_KEY=your_api_key_here
   AZURE_ENDPOINT=https://your-resource.openai.azure.com
   ALLOWED_USERS=123456789,987654321
   RENDER=true
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait for deployment to complete

**Cost:** Free plan available (with limitations)

---

### 3. Railway.app

**Step-by-step:**

1. **Create Railway account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

2. **New project from GitHub**
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository

3. **Environment variables:**
   - Go to "Variables" tab
   - Add the same variables as Render

4. **Auto-deploy**
   - Railway auto-detects `requirements.txt`
   - Bot deploys automatically

**Cost:** $5/month free credit, then pay-as-you-go

---

### 4. VPS (DigitalOcean, Linode, etc.)

**Step-by-step:**

1. **Create droplet/server**
   - Ubuntu 22.04 LTS recommended
   - Minimum 1GB RAM

2. **Connect via SSH**
   ```bash
   ssh root@your-ip
   ```

3. **Install Python and dependencies**
   ```bash
   apt update
   apt install python3.10 python3-pip git -y
   ```

4. **Clone repository**
   ```bash
   git clone https://github.com/your-user/tg_Cortana_GPT_bot.git
   cd tg_Cortana_GPT_bot
   ```

5. **Configure environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

6. **Create .env file**
   ```bash
   nano .env
   # Paste your environment variables
   # Ctrl+X, Y, Enter to save
   ```

7. **Run with systemd (auto-restart)**
   
   Create service file:
   ```bash
   nano /etc/systemd/system/telegram-bot.service
   ```
   
   Content:
   ```ini
   [Unit]
   Description=Telegram Bot GPT-4o
   After=network.target

   [Service]
   Type=simple
   User=root
   WorkingDirectory=/root/tg_Cortana_GPT_bot
   Environment="PATH=/root/tg_Cortana_GPT_bot/venv/bin"
   ExecStart=/root/tg_Cortana_GPT_bot/venv/bin/python bot.py
   Restart=always
   RestartSec=10

   [Install]
   WantedBy=multi-user.target
   ```
   
   Enable service:
   ```bash
   systemctl daemon-reload
   systemctl enable telegram-bot
   systemctl start telegram-bot
   systemctl status telegram-bot
   ```

**Cost:** From $5/month

---

## Security Checklist

Before deploying to production:

- [ ] ✅ `ALLOWED_USERS` configured with valid IDs
- [ ] ✅ Environment variables NOT in code
- [ ] ✅ `.env` is in `.gitignore`
- [ ] ✅ Azure API keys are valid
- [ ] ✅ Telegram bot token is valid
- [ ] ✅ Logs are configured correctly
- [ ] ⚠️ Consider adding rate limiting
- [ ] ⚠️ Consider using database for persistence

---

## Monitoring

### Logs on Render/Railway

Both platforms have dashboards with real-time logs:
- Render: "Logs" tab
- Railway: "Deployments" tab → Click on deployment

### Logs on VPS

```bash
# View logs in real-time
journalctl -u telegram-bot -f

# View last 100 logs
journalctl -u telegram-bot -n 100

# View today's logs
journalctl -u telegram-bot --since today
```

---

## Updating the Bot

### On Render/Railway (with GitHub)

1. Push your changes to GitHub
2. Bot redeploys automatically

### On VPS

```bash
cd /root/tg_Cortana_GPT_bot
git pull
systemctl restart telegram-bot
```

---

## Production Troubleshooting

### Bot doesn't respond

1. **Verify it's running:**
   ```bash
   # Render/Railway: Check logs in dashboard
   # VPS:
   systemctl status telegram-bot
   ```

2. **Verify environment variables:**
   - Render/Railway: "Environment" tab
   - VPS: `cat .env`

3. **Check logs:**
   - Look for authentication errors
   - Verify ALLOWED_USERS is configured

### Out of Memory (OOM)

If bot runs out of memory:
- Render: Upgrade to Starter plan ($7/month) with 512MB RAM
- Railway: Adjust memory limits
- VPS: Upgrade to plan with more RAM

### Telegram Rate Limiting

If you receive "Too Many Requests" errors:
- Telegram limits to ~30 messages/second
- Add delays between bulk messages
- Consider using `asyncio.sleep()` between calls

---

## Future Improvements

To scale the bot in production:

1. **Database:**
   - PostgreSQL for conversations
   - Redis for cache

2. **Rate limiting:**
   - Limit messages per user
   - Prevent abuse

3. **Metrics:**
   - Prometheus + Grafana
   - Token usage monitoring

4. **Backup:**
   - Automatic conversation backup
   - Disaster recovery plan
