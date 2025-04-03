# filepath: bot.py
# Main bot file for Telegram bot with ChatGPT and DALL-E integration
# Optimized for both local development and deployment on Render.com

import logging
import os
import time
import json
import sys
import uuid
from datetime import datetime
from collections import defaultdict, deque
from functools import wraps

# Third-party imports
from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram.ext import Dispatcher, CallbackQueryHandler
import openai
from dotenv import load_dotenv
from flask import Flask, request, Response

# Load environment variables from .env file (for local development)
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO")
log_level_dict = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL
}

# In production (Render.com), we'll log to stdout only
# In development, we'll also log to a file
log_handlers = [logging.StreamHandler()]
if os.getenv("RENDER") != "true":
    log_handlers.append(logging.FileHandler("bot.log"))

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=log_level_dict.get(log_level, logging.INFO),
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

# Configure API keys from environment variables
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
AZURE_API_VERSION = os.getenv(
    "AZURE_API_VERSION", "2023-05-15")  # Default API version

# Configure limits
MAX_TOKENS_PER_USER = int(os.getenv("MAX_TOKENS_PER_USER", 1000))
MAX_CONVERSATIONS_STORED = int(os.getenv("MAX_CONVERSATIONS_STORED", 10))

# Configure webhook settings for Render.com
PORT = int(os.getenv("PORT", 8080))
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
IS_PRODUCTION = os.getenv("RENDER", "false").lower() == "true"

# Generate a random path for webhook to improve security
WEBHOOK_PATH = os.getenv("WEBHOOK_PATH", str(uuid.uuid4()))

# Check if required environment variables are set
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, TELEGRAM_BOT_TOKEN]):
    logger.critical(
        "Missing required environment variables. Please configure the .env file")
    exit(1)

# Configure OpenAI for Azure
openai.api_type = "azure"
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = AZURE_API_VERSION

# Structure to store token usage per user
user_token_usage = defaultdict(int)

# Structure to store conversations per user
user_conversations = defaultdict(
    lambda: deque(maxlen=MAX_CONVERSATIONS_STORED))

# Simple response cache to avoid duplicate API calls
response_cache = {}

# Dictionary of available models with their costs per 1K tokens and deployment IDs
models = {
    "gpt-4": {
        "name": "gpt-4",
        "deployment_id": "gpt-4",  # Update with your actual deployment ID
        "input_cost": 0.03,
        "output_cost": 0.06
    },
    "gpt-4o": {
        "name": "gpt-4o",
        "deployment_id": "gpt-4o",  # Update with your actual deployment ID
        "input_cost": 0.01,
        "output_cost": 0.03
    },
    "gpt-4-32k": {
        "name": "gpt-4-32k",
        "deployment_id": "gpt-4-32k",  # Update with your actual deployment ID
        "input_cost": 0.06,
        "output_cost": 0.12
    },
    "gpt-35-turbo": {
        "name": "gpt-35-turbo",
        "deployment_id": "gpt-35-turbo",  # Update with your actual deployment ID
        "input_cost": 0.001,
        "output_cost": 0.002
    },
    "gpt-35-turbo-16k": {
        "name": "gpt-35-turbo-16k",
        "deployment_id": "gpt-35-turbo-16k",  # Update with your actual deployment ID
        "input_cost": 0.003,
        "output_cost": 0.004
    },
    "text-embedding-ada-002": {
        "name": "text-embedding-ada-002",
        # Update with your actual deployment ID
        "deployment_id": "text-embedding-ada-002",
        "input_cost": 0.0001,
        "output_cost": 0
    },
    "dall-e-3": {
        "name": "dall-e-3",
        "deployment_id": "dall-e-3",  # Update with your actual deployment ID
        "input_cost": 0,
        "output_cost": 0
    }
}

# Data persistence functions


def save_user_data():
    """Save user data to a file for persistence"""
    data = {
        "token_usage": {str(k): v for k, v in user_token_usage.items()},
        "conversations": {str(k): list(v) for k, v in user_conversations.items()}
    }
    try:
        with open('user_data.json', 'w') as f:
            json.dump(data, f)
        logger.info("User data saved successfully")
    except Exception as e:
        logger.error(f"Failed to save user data: {e}")


def load_user_data():
    """Load user data from a file if it exists"""
    if os.path.exists('user_data.json'):
        try:
            with open('user_data.json', 'r') as f:
                data = json.load(f)
                for k, v in data.get("token_usage", {}).items():
                    user_token_usage[int(k)] = v
                for k, v in data.get("conversations", {}).items():
                    user_conversations[int(k)] = deque(
                        v, maxlen=MAX_CONVERSATIONS_STORED)
            logger.info("User data loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load user data: {e}")

# Decorator to track usage and check limits


def track_usage(func):
    @wraps(func)
    def wrapper(update: Update, context: CallbackContext, *args, **kwargs):
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"

        # Check if the user has exceeded their limit
        if user_token_usage[user_id] >= MAX_TOKENS_PER_USER:
            logger.warning(
                f"User {username} (ID: {user_id}) has exceeded their token limit")
            update.message.reply_text(
                "You have reached your usage limit. Please try again later or contact the administrator."
            )
            return

        # Log the request
        logger.info(
            f"Request from {username} (ID: {user_id}): {update.message.text}")

        # Execute the original function
        return func(update, context, *args, **kwargs)

    return wrapper

# Improved retry decorator with exponential backoff


def retry_on_error(max_retries=3, backoff_factor=1.5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = 1  # Start with 1 second delay
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    logger.warning(
                        f"Attempt {retries}/{max_retries} failed: {e}")
                    if retries >= max_retries:
                        logger.error(
                            f"Error after {max_retries} attempts: {e}")
                        raise
                    # Exponential backoff
                    time.sleep(delay)
                    delay *= backoff_factor
            return None
        return wrapper
    return decorator


def start(update: Update, context: CallbackContext):
    """Handler for the /start command"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    logger.info(f"User {username} (ID: {user_id}) has started the bot")

    # Initialize user data if it's the first time
    if "model" not in context.user_data:
        context.user_data["model"] = "gpt-35-turbo"

    model_names = [model_info["name"] for model_info in models.values()]

    update.message.reply_text(
        f"Hello {update.effective_user.first_name}! I'm a bot powered by ChatGPT. Send me a message and I'll respond.\n\n"
        "*Available commands:*\n"
        "/start - Show this help message\n"
        "/help - Show detailed information about commands\n"
        "/setmodel <model_name> - Choose the model you want to work with\n"
        "/generatehd <description> - Generate an HD image (1920x1080)\n"
        "/reset - Reset the current conversation\n"
        "/usage - Show your current token usage\n\n"
        f"*Current model:* {models[context.user_data['model']]['name']}\n"
        f"*Available models:* {', '.join(model_names)}",
        parse_mode=ParseMode.MARKDOWN
    )


@track_usage
def set_model(update: Update, context: CallbackContext):
    """Handler for the /setmodel command"""
    if len(context.args) == 0:
        model_names = [model_info["name"] for model_info in models.values()]
        update.message.reply_text(
            "Please provide the model name.\n"
            f"Example: /setmodel gpt-4\n\n"
            f"Available models: {', '.join(model_names)}"
        )
        return

    model_name = " ".join(context.args)
    model_key = None

    for key, value in models.items():
        if value["name"].lower() == model_name.lower():
            model_key = key
            break

    if model_key:
        old_model = context.user_data.get("model", "gpt-35-turbo")
        context.user_data["model"] = model_key
        logger.info(
            f"User {update.effective_user.id} changed model from {old_model} to {model_key}")
        update.message.reply_text(
            f"✅ Model changed to *{models[model_key]['name']}*\n\n"
            f"*Approximate costs per 1K tokens:*\n"
            f"- Input: ${models[model_key]['input_cost']}\n"
            f"- Output: ${models[model_key]['output_cost']}",
            parse_mode=ParseMode.MARKDOWN
        )
    else:
        model_names = [model_info["name"] for model_info in models.values()]
        update.message.reply_text(
            f"❌ Model not found.\n\n"
            f"*Available models:* {', '.join(model_names)}",
            parse_mode=ParseMode.MARKDOWN
        )


@track_usage
@retry_on_error(max_retries=2, backoff_factor=2)
def generate_image(update: Update, context: CallbackContext, size="1024x1024"):
    """Handler to generate images using DALL-E"""
    if len(context.args) == 0:
        update.message.reply_text(
            "Please provide a description for the image.\n"
            "Example: /generatehd A mountain landscape at sunset"
        )
        return

    user_message = " ".join(context.args)
    user_id = update.effective_user.id

    # Log the image request
    logger.info(f"User {user_id} requested to generate image: {user_message}")

    # Check prompt length
    if len(user_message) > 1000:
        update.message.reply_text(
            "The description is too long. Please limit it to 1000 characters.")
        return

    # Wait message with emoji for better UX
    wait_message = update.message.reply_text(
        "🎨 Generating image... Please wait.")

    try:
        # Check cache first
        cache_key = f"image:{user_message}:{size}"
        if cache_key in response_cache:
            image_url = response_cache[cache_key]
            logger.info(f"Returning cached image for user {user_id}")
        else:
            # Increment usage counter (estimation for images)
            user_token_usage[user_id] += 500

            # For DALL-E in Azure, we need to use a different API call
            # This is specifically for Azure OpenAI's DALL-E implementation
            response = openai.Image.create(
                api_version="2023-06-01-preview",  # Specific version for DALL-E in Azure
                prompt=user_message,
                n=1,
                size=size,
                deployment_id=models["dall-e-3"]["deployment_id"]
            )

            # Parse the response to get the image URL
            # The structure might differ between Azure and standard OpenAI
            if 'data' in response and len(response['data']) > 0:
                if 'url' in response['data'][0]:
                    image_url = response['data'][0]['url']
                else:
                    # Some versions return b64_json instead of url
                    image_url = "Image generated but URL not available. Check Azure portal."
            else:
                raise Exception(
                    "Unexpected response format from Azure DALL-E API")

            # Cache the result
            response_cache[cache_key] = image_url

        # Save the request in history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_conversations[user_id].append({
            "type": "image",
            "prompt": user_message,
            "url": image_url,
            "timestamp": timestamp
        })

        # Save updated user data
        save_user_data()

        # Delete wait message and send the image
        wait_message.delete()
        update.message.reply_text(
            f"✨ *Image generated successfully*\n\n"
            f"*Prompt:* {user_message}\n\n"
            f"[View image]({image_url})",
            parse_mode=ParseMode.MARKDOWN
        )

        logger.info(f"Image generated successfully for user {user_id}")
    except Exception as e:
        wait_message.delete()
        error_message = str(e)
        logger.error(
            f"Error generating image for user {user_id}: {error_message}")

        # More detailed error message
        if "content policy violation" in error_message.lower():
            update.message.reply_text(
                "❌ Could not generate the image because the requested content violates usage policies.\n"
                "Please try with a different description."
            )
        else:
            update.message.reply_text(
                f"❌ Sorry, an error occurred while generating the image.\n"
                f"Error: {error_message[:100]}..."
            )


@track_usage
@retry_on_error(max_retries=2, backoff_factor=2)
def chat(update: Update, context: CallbackContext):
    """Handler for processing chat messages and generating AI responses"""
    user_id = update.effective_user.id
    user_message = update.message.text
    model_key = context.user_data.get("model", "gpt-35-turbo")  # Default model

    # Initialize conversation if it doesn't exist
    if "messages" not in context.user_data:
        context.user_data["messages"] = []

    # Add the user message to history
    context.user_data["messages"].append(
        {"role": "user", "content": user_message})

    # Limit the number of messages to avoid excessive tokens
    if len(context.user_data["messages"]) > 20:  # Adjust as needed
        context.user_data["messages"] = context.user_data["messages"][-20:]

    # Wait message with typing indicator
    wait_message = update.message.reply_text("💭 Thinking...")

    try:
        # Check cache first
        messages_str = json.dumps(context.user_data["messages"])
        cache_key = f"chat:{model_key}:{messages_str}"

        if cache_key in response_cache:
            bot_reply = response_cache[cache_key]
            logger.info(f"Using cached response for user {user_id}")
            estimated_tokens = len(bot_reply) // 4
        else:
            # Prepare messages for the API including conversation history
            messages = context.user_data["messages"]

            # Estimate input tokens (approximate)
            # ~4 characters per token
            input_tokens = sum(len(msg["content"]) // 4 for msg in messages)

            # Check token limit
            if user_token_usage[user_id] + input_tokens > MAX_TOKENS_PER_USER:
                wait_message.delete()
                update.message.reply_text(
                    "You have reached your usage limit. Please try again later or contact the administrator."
                )
                return

            # Call the API with Azure-specific parameters
            response = openai.ChatCompletion.create(
                # Use deployment_id for Azure
                deployment_id=models[model_key]["deployment_id"],
                messages=messages
            )

            # Extract response
            bot_reply = response.choices[0].message.content

            # Update token usage with actual values from API
            total_tokens = response.usage.total_tokens if hasattr(
                response, 'usage') else input_tokens + (len(bot_reply) // 4)
            user_token_usage[user_id] += total_tokens

            # Cache the response
            response_cache[cache_key] = bot_reply

            # Log total usage
            logger.info(
                f"User {user_id} used {total_tokens} tokens with model {model_key}")

        # Add response to conversation history
        context.user_data["messages"].append(
            {"role": "assistant", "content": bot_reply})

        # Save in general history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_conversations[user_id].append({
            "type": "chat",
            "user_message": user_message,
            "bot_reply": bot_reply,
            "model": model_key,
            "tokens": total_tokens if 'total_tokens' in locals() else len(bot_reply) // 4,
            "timestamp": timestamp
        })

        # Save updated user data
        save_user_data()

        # Delete wait message and send response
        wait_message.delete()
        update.message.reply_text(bot_reply)

    except Exception as e:
        wait_message.delete()
        error_message = str(e)
        logger.error(
            f"Error generating response for user {user_id}: {error_message}")

        # More detailed error message
        if "maximum context length" in error_message.lower():
            # Reset context if it's too long
            context.user_data["messages"] = context.user_data["messages"][-2:]
            update.message.reply_text(
                "❌ The conversation is too long. I've reset the context.\n"
                "Please try sending your message again."
            )
        else:
            update.message.reply_text(
                f"❌ Sorry, an error occurred while generating the response.\n"
                f"Error: {error_message[:100]}..."
            )


# Command to show current usage
@track_usage
def show_usage(update: Update, context: CallbackContext):
    """Handler for the /usage command to display token usage statistics"""
    user_id = update.effective_user.id
    tokens_used = user_token_usage[user_id]
    percentage = (tokens_used / MAX_TOKENS_PER_USER) * 100

    # Get current model
    model_key = context.user_data.get("model", "gpt-35-turbo")
    model_info = models[model_key]

    # Calculate approximate cost
    # Assuming half are input tokens
    input_cost = (tokens_used / 2) * (model_info["input_cost"] / 1000)
    # Assuming half are output tokens
    output_cost = (tokens_used / 2) * (model_info["output_cost"] / 1000)
    total_cost = input_cost + output_cost

    update.message.reply_text(
        f"📊 *Usage Statistics*\n\n"
        f"*Tokens used:* {tokens_used} of {MAX_TOKENS_PER_USER} ({percentage:.1f}%)\n"
        f"*Current model:* {model_info['name']}\n"
        f"*Approximate cost:* ${total_cost:.4f}\n\n"
        f"*Conversations saved:* {len(user_conversations[user_id])}\n"
        f"*Conversation limit:* {MAX_CONVERSATIONS_STORED}",
        parse_mode=ParseMode.MARKDOWN
    )

# Command to reset the conversation


@track_usage
def reset_conversation(update: Update, context: CallbackContext):
    """Handler for the /reset command to clear conversation history"""
    user_id = update.effective_user.id

    if "messages" in context.user_data:
        # Save the conversation before resetting it
        if context.user_data["messages"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_conversations[user_id].append({
                "type": "full_conversation",
                "messages": context.user_data["messages"].copy(),
                "timestamp": timestamp
            })

        # Reset the conversation
        context.user_data["messages"] = []
        logger.info(f"User {user_id} has reset their conversation")
        update.message.reply_text(
            "🔄 Conversation reset. Let's start fresh!")

        # Save updated user data
        save_user_data()
    else:
        update.message.reply_text(
            "There is no active conversation to reset.")

# Detailed help command


@track_usage
def help_command(update: Update, context: CallbackContext):
    """Handler for the /help command with detailed usage information"""
    update.message.reply_text(
        "*🤖 Bot Usage Guide*\n\n"
        "*Available commands:*\n\n"
        "*/start* - Start the bot and show the welcome message\n"
        "*/help* - Show this detailed help guide\n"
        "*/setmodel <name>* - Change the AI model to use\n"
        "*/generatehd <description>* - Generate a high-definition image (1024x1024)\n"
        "*/reset* - Reset the current conversation, removing the context\n"
        "*/usage* - Show your current usage statistics\n\n"
        "*Tips:*\n"
        "• Be clear and specific in your questions for better responses\n"
        "• The bot maintains conversation context, so you can ask follow-up questions\n"
        "• If the bot seems confused, use /reset to start a new conversation\n"
        "• Use detailed descriptions for better image generation results\n\n"
        "*Available models:*\n"
        "• gpt-4 - Most advanced, best for complex tasks\n"
        "• gpt-4o - Optimized version of GPT-4\n"
        "• gpt-35-turbo - Fast and efficient for most tasks\n"
        "• dall-e-3 - For image generation\n\n"
        "If you have any issues or suggestions, please contact the bot administrator.",
        parse_mode=ParseMode.MARKDOWN
    )

# Function to create Flask app (for Gunicorn)


def create_app():
    """Create and configure the Flask application for webhook handling"""
    global bot, dispatcher

    # Initialize bot and dispatcher
    bot = Updater(TELEGRAM_BOT_TOKEN).bot
    dispatcher = setup_dispatcher()

    # Set up the Flask app
    app = Flask(__name__)

    # Define webhook route with secure path
    @app.route(f'/{WEBHOOK_PATH}', methods=['POST'])
    def webhook():
        """Handle incoming updates from Telegram"""
        update = Update.de_json(request.get_json(force=True), bot)
        dispatcher.process_update(update)
        return 'ok'

    @app.route('/')
    def index():
        """Health check endpoint"""
        return 'Bot is running!'

    return app


def setup_dispatcher():
    """Setup and return the dispatcher with all handlers"""
    dispatcher = Dispatcher(bot, None, workers=0)

    # Register command handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler(
        "setmodel", set_model, pass_args=True))
    dispatcher.add_handler(CommandHandler(
        "generatehd", generate_image, pass_args=True))
    dispatcher.add_handler(CommandHandler("reset", reset_conversation))
    dispatcher.add_handler(CommandHandler("usage", show_usage))
    dispatcher.add_handler(MessageHandler(
        Filters.text & ~Filters.command, chat))

    # Register error handler
    dispatcher.add_error_handler(lambda update, context: logger.error(
        f"Error in update {update}: {context.error}"))

    return dispatcher


def main():
    """Main function to run the bot"""
    # Load saved user data if available
    load_user_data()

    # Verify that API keys are configured
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, TELEGRAM_BOT_TOKEN]):
        logger.critical(
            "Missing required environment variables. Please configure the .env file")
        print(
            "ERROR: Missing required environment variables. Please configure the .env file")
        return

    global bot, dispatcher

    try:
        logger.info("Starting the bot...")
        bot = Updater(TELEGRAM_BOT_TOKEN).bot
        dispatcher = setup_dispatcher()

        # Choose between webhook (production) and polling (development)
        if IS_PRODUCTION and WEBHOOK_URL:
            # Production mode: use webhook
            logger.info(f"Starting webhook on port {PORT}")

            # Set webhook with secure path
            webhook_url = f"{WEBHOOK_URL}/{WEBHOOK_PATH}"
            bot.set_webhook(url=webhook_url)
            logger.info(f"Webhook set to {webhook_url}")

            # Start Flask server
            logger.info("Bot started successfully in webhook mode")
            app = create_app()
            app.run(host='0.0.0.0', port=PORT)
        else:
            # Development mode: use polling
            logger.info("Starting polling")
            updater = Updater(TELEGRAM_BOT_TOKEN)
            updater.dispatcher = dispatcher

            # Start the Bot in polling mode
            logger.info("Bot started successfully in polling mode")
            print("Bot started successfully. Press Ctrl+C to stop.")
            updater.start_polling()
            updater.idle()
    except Exception as e:
        logger.critical(f"Error starting the bot: {e}")
        print(f"ERROR: Could not start the bot: {e}")


if __name__ == "__main__":
    main()
