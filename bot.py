#!/usr/bin/env python
"""
Telegram Bot with GPT-4o and DALL-E 3 Integration
Production-ready bot with async architecture and security features
"""

import logging
import os
import json
import asyncio
from enum import Enum
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any
from functools import wraps
from http.server import BaseHTTPRequestHandler, HTTPServer

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)

from openai import AsyncAzureOpenAI

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger("TelegramBot")

# Environment variables
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
AZURE_API_KEY = os.getenv('AZURE_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_ENDPOINT')
AZURE_API_VERSION = os.getenv('AZURE_API_VERSION')

# Security: Allowed Telegram user IDs
ALLOWED_USERS_STR = os.getenv('ALLOWED_USERS', '')
ALLOWED_USERS = [int(id.strip()) for id in ALLOWED_USERS_STR.split(',') if id.strip()]

# Validate required environment variables
if not all([TELEGRAM_BOT_TOKEN, AZURE_API_KEY, AZURE_ENDPOINT]):
    raise EnvironmentError("Missing required environment variables: TELEGRAM_BOT_TOKEN, AZURE_API_KEY, AZURE_ENDPOINT")

logger.info(f"Using Azure API version: {AZURE_API_VERSION}")

if not ALLOWED_USERS:
    logger.warning("⚠️ ALLOWED_USERS is empty - bot will allow access to ALL users!")
    logger.warning("⚠️ This is NOT recommended for cloud deployment!")

# Azure OpenAI client
client = AsyncAzureOpenAI(
    api_key=AZURE_API_KEY,
    api_version=AZURE_API_VERSION,
    azure_endpoint=AZURE_ENDPOINT
)

# Tools schema for GPT-4o function calling
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image based on a detailed description using DALL-E 3. Use when user explicitly requests to draw, paint, create an image or photo.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "Detailed description of the image to generate, translated to English for best results."
                    },
                    "quality": {
                        "type": "string",
                        "enum": ["standard", "hd"],
                        "description": "Image quality. Default is standard."
                    }
                },
                "required": ["prompt"]
            }
        }
    }
]


class AzureDeployment(Enum):
    """Azure OpenAI deployment names"""
    GPT4O = "gpt-4o"
    DALLE3 = "dall-e-3"


class HealthCheckHandler(BaseHTTPRequestHandler):
    """HTTP handler for Render.com health checks"""
    
    def do_GET(self):
        """Handle GET requests for health check"""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Bot is running")
    
    def log_message(self, format, *args):
        """Suppress HTTP server logs"""
        pass


def start_health_server():
    """Start HTTP server for Render.com Web Service compatibility"""
    port = int(os.getenv('PORT', 8080))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    logger.info(f"🌐 Health check server started on port {port}")
    server.serve_forever()


class ConversationManager:
    """Manages conversation history per user"""
    
    def __init__(self, max_history: int = 15):
        self.conversations: Dict[int, List[Dict]] = {}
        self.max_history = max_history

    def add_message(self, user_id: int, role: str, content: str, tool_calls: List = None, name: str = None):
        """Add a message to user's conversation history"""
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        msg = {"role": role, "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if name:
            msg["name"] = name
            
        self.conversations[user_id].append(msg)
        
        # Trim history to prevent context overflow
        if len(self.conversations[user_id]) > self.max_history:
            self.conversations[user_id] = self.conversations[user_id][-self.max_history:]

    def get_history(self, user_id: int) -> List[Dict]:
        """Get user's conversation history"""
        return self.conversations.get(user_id, [])

    def clear(self, user_id: int):
        """Clear user's conversation history"""
        self.conversations[user_id] = []


conv_manager = ConversationManager()


def restricted(func):
    """Decorator to restrict access to allowed users only"""
    @wraps(func)
    async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        if ALLOWED_USERS and user_id not in ALLOWED_USERS:
            logger.warning(f"⛔ Access denied for user {username} (ID: {user_id})")
            await update.message.reply_text("⛔ You don't have permission to use this bot.")
            return
        
        logger.info(f"✅ Authorized user: {username} (ID: {user_id})")
        return await func(update, context, *args, **kwargs)
    return wrapped


async def generate_dalle_image(prompt: str, quality: str = "standard") -> str:
    """Generate image using DALL-E 3"""
    try:
        logger.info(f"🎨 DALL-E Prompt: {prompt} | Quality: {quality}")
        response = await client.images.generate(
            model=AzureDeployment.DALLE3.value,
            prompt=prompt,
            size="1024x1024",
            quality=quality,
            n=1
        )
        return response.data[0].url
    except Exception as e:
        logger.error(f"❌ Error generating image: {e}")
        return None


@restricted
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    welcome_msg = (
        "👋 **Hello! I'm your GPT-4o assistant.**\n\n"
        "You can talk to me normally or ask me to generate images.\n\n"
        "**Examples:**\n"
        "- *'Explain the theory of relativity'*\n"
        "- *'Generate an image of a cyberpunk cat'*\n\n"
        "**Commands:**\n"
        "/start - Show this message\n"
        "/clear - Clear conversation memory"
    )
    await update.message.reply_text(welcome_msg, parse_mode='Markdown')


@restricted
async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /clear command"""
    user_id = update.effective_user.id
    conv_manager.clear(user_id)
    logger.info(f"🧹 User {user_id} cleared conversation history")
    await update.message.reply_text("🧹 Memory cleared. Starting fresh.")


@restricted
async def chat_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle chat messages with GPT-4o and function calling"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    user_text = update.message.text
    
    logger.info(f"💬 Message from {username} (ID: {user_id}): {user_text}")
    
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)

    conv_manager.add_message(user_id, "user", user_text)
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful, friendly, and creative assistant. If asked for an image, use the available tool. Respond in the same language the user speaks to you."
        },
        *conv_manager.get_history(user_id)
    ]

    try:
        response = await client.chat.completions.create(
            model=AzureDeployment.GPT4O.value,
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
            temperature=0.7
        )
        
        response_msg = response.choices[0].message
        
        # Handle tool calls (image generation)
        if response_msg.tool_calls:
            logger.info(f"🔧 GPT-4o requested tool: {response_msg.tool_calls[0].function.name}")
            
            await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
            
            tool_call = response_msg.tool_calls[0]
            if tool_call.function.name == "generate_image":
                args = json.loads(tool_call.function.arguments)
                prompt = args.get("prompt")
                quality = args.get("quality", "standard")
                
                await update.message.reply_text(f"🎨 Generating image: {prompt}...")
                
                image_url = await generate_dalle_image(prompt, quality)
                
                if image_url:
                    await update.message.reply_photo(photo=image_url, caption=f"✨ {prompt}")
                    logger.info(f"✅ Image generated successfully for user {user_id}")
                    
                    conv_manager.add_message(
                        user_id, 
                        "assistant", 
                        None, 
                        tool_calls=[{
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments
                            }
                        }]
                    )
                    conv_manager.add_message(
                        user_id, 
                        "tool", 
                        "Image generated and sent successfully.", 
                        name=tool_call.function.name
                    )
                else:
                    await update.message.reply_text("❌ Error generating image.")
                    logger.error(f"❌ Image generation failed for user {user_id}")
        
        # Handle text responses
        else:
            bot_reply = response_msg.content
            conv_manager.add_message(user_id, "assistant", bot_reply)
            await update.message.reply_text(bot_reply)
            logger.info(f"✅ Response sent to user {user_id}")

    except Exception as e:
        logger.error(f"❌ Error in chat handler for user {user_id}: {e}")
        await update.message.reply_text(
            "😵 An error occurred processing your request. Please try again."
        )


def main():
    """Initialize and run the bot"""
    print("🚀 Bot starting...")
    logger.info("🚀 Starting Telegram bot with GPT-4o and DALL-E 3")
    
    # Start health check server for Render.com if in production
    is_render = os.getenv('RENDER', 'false').lower() == 'true'
    if is_render:
        logger.info("🌐 Running on Render.com - starting health check server")
        Thread(target=start_health_server, daemon=True).start()
    
    if ALLOWED_USERS:
        logger.info(f"🔒 Security mode: {len(ALLOWED_USERS)} authorized user(s)")
    else:
        logger.warning("⚠️ Open mode: All users can access the bot")
    
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat_handler))

    logger.info("✅ Bot ready - Polling mode activated")
    print("✅ Bot ready and listening for messages...")
    application.run_polling()


if __name__ == "__main__":
    main()
