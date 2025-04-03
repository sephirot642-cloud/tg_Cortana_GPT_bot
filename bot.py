# filepath: /C:/Users/Sephirot/OneDrive/Documentos/Coding Projects/tg_Cortana_GPT_bot/bot.py
# Main bot file for Telegram bot with ChatGPT and DALL-E integration
# Modified for deployment on Render.com

from telegram import Update, ParseMode
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext
from telegram.ext import Dispatcher, CallbackQueryHandler
import openai
import logging
import os
import time
import json
import sys
from datetime import datetime
from collections import defaultdict, deque
from functools import wraps
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

# Configure limits
MAX_TOKENS_PER_USER = int(os.getenv("MAX_TOKENS_PER_USER", 1000))
MAX_CONVERSATIONS_STORED = int(os.getenv("MAX_CONVERSATIONS_STORED", 10))

# Configure webhook settings for Render.com
PORT = int(os.getenv("PORT", 8080))
WEBHOOK_URL = os.getenv("WEBHOOK_URL")
IS_PRODUCTION = os.getenv("RENDER", "false").lower() == "true"

# Create Flask app for webhook handling
app = Flask(__name__)

# Verificar que las variables de entorno necesarias estén configuradas
if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, TELEGRAM_BOT_TOKEN]):
    logger.critical("Faltan variables de entorno necesarias. Por favor, configura el archivo .env")
    exit(1)

openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT

# Estructura para almacenar el uso de tokens por usuario
user_token_usage = defaultdict(int)

# Estructura para almacenar conversaciones por usuario
user_conversations = defaultdict(lambda: deque(maxlen=MAX_CONVERSATIONS_STORED))

# Diccionario de modelos disponibles con sus costos aproximados por 1K tokens
models = {
    "gpt-4": {"name": "gpt-4", "input_cost": 0.03, "output_cost": 0.06},
    "gpt-4o": {"name": "gpt-4o", "input_cost": 0.01, "output_cost": 0.03},
    "gpt-4-32k": {"name": "gpt-4-32k", "input_cost": 0.06, "output_cost": 0.12},
    "gpt-35-turbo": {"name": "gpt-35-turbo", "input_cost": 0.001, "output_cost": 0.002},
    "gpt-35-turbo-16k": {"name": "gpt-35-turbo-16k", "input_cost": 0.003, "output_cost": 0.004},
    "text-embedding-ada-002": {"name": "text-embedding-ada-002", "input_cost": 0.0001, "output_cost": 0},
    "dall-e-3": {"name": "dall-e-3", "input_cost": 0, "output_cost": 0}
}

# Decorador para registrar el uso y verificar límites
def track_usage(func):
    @wraps(func)
    def wrapper(update: Update, context: CallbackContext, *args, **kwargs):
        user_id = update.effective_user.id
        username = update.effective_user.username or "Unknown"
        
        # Verificar si el usuario ha excedido su límite
        if user_token_usage[user_id] >= MAX_TOKENS_PER_USER:
            logger.warning(f"Usuario {username} (ID: {user_id}) ha excedido su límite de tokens")
            update.message.reply_text(
                "Has alcanzado tu límite de uso. Por favor, intenta más tarde o contacta al administrador."
            )
            return
        
        # Registrar la solicitud
        logger.info(f"Solicitud de {username} (ID: {user_id}): {update.message.text}")
        
        # Ejecutar la función original
        return func(update, context, *args, **kwargs)
    
    return wrapper

# Función para reintentar en caso de error
def retry_on_error(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            while retries < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    logger.warning(f"Intento {retries}/{max_retries} falló: {e}")
                    if retries >= max_retries:
                        logger.error(f"Error después de {max_retries} intentos: {e}")
                        raise
                    time.sleep(delay)
            return None
        return wrapper
    return decorator


def start(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    logger.info(f"Usuario {username} (ID: {user_id}) ha iniciado el bot")
    
    # Inicializar datos del usuario si es la primera vez
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
        logger.info(f"User {update.effective_user.id} changed model from {old_model} to {model_key}")
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
@retry_on_error(max_retries=2)
def generate_image(update: Update, context: CallbackContext, size="1920x1080"):
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
        update.message.reply_text("The description is too long. Please limit it to 1000 characters.")
        return
    
    # Wait message with emoji for better UX
    wait_message = update.message.reply_text("🎨 Generating image... Please wait.")
    
    try:
        # Increment usage counter (estimation for images)
        user_token_usage[user_id] += 500
        
        response = openai.Image.create(
            prompt=user_message,
            n=1,
            size=size
        )
        image_url = response['data'][0]['url']
        
        # Save the request in history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_conversations[user_id].append({
            "type": "image",
            "prompt": user_message,
            "url": image_url,
            "timestamp": timestamp
        })
        
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
        logger.error(f"Error generating image for user {user_id}: {error_message}")
        
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
@retry_on_error(max_retries=2)
def chat(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    user_message = update.message.text
    model_key = context.user_data.get("model", "gpt-35-turbo")  # Modelo predeterminado
    
    # Inicializar la conversación si no existe
    if "messages" not in context.user_data:
        context.user_data["messages"] = []
    
    # Añadir el mensaje del usuario al historial
    context.user_data["messages"].append({"role": "user", "content": user_message})
    
    # Limitar el número de mensajes para evitar tokens excesivos (mantener contexto reciente)
    if len(context.user_data["messages"]) > 20:  # Ajustar según necesidades
        context.user_data["messages"] = context.user_data["messages"][-20:]
    
    # Mensaje de espera con indicador de escritura
    wait_message = update.message.reply_text("💭 Pensando...")  
    
    try:
        # Preparar mensajes para la API incluyendo el historial de conversación
        messages = context.user_data["messages"]
        
        # Estimar tokens de entrada (aproximado)
        input_tokens = sum(len(msg["content"]) // 4 for msg in messages)  # ~4 caracteres por token
        
        # Verificar límite de tokens
        if user_token_usage[user_id] + input_tokens > MAX_TOKENS_PER_USER:
            wait_message.delete()
            update.message.reply_text(
                "Has alcanzado tu límite de uso. Por favor, intenta más tarde o contacta al administrador."
            )
            return
        
        # Incrementar contador de tokens (estimación)
        user_token_usage[user_id] += input_tokens
        
        # Llamar a la API
        response = openai.ChatCompletion.create(
            model=model_key,
            messages=messages
        )
        
        # Extraer respuesta
        bot_reply = response["choices"][0]["message"]["content"]
        
        # Estimar tokens de salida y actualizar contador
        output_tokens = len(bot_reply) // 4  # ~4 caracteres por token
        user_token_usage[user_id] += output_tokens
        
        # Registrar uso total
        total_tokens = response.get("usage", {}).get("total_tokens", input_tokens + output_tokens)
        logger.info(f"Usuario {user_id} usó {total_tokens} tokens con modelo {model_key}")
        
        # Añadir respuesta al historial
        context.user_data["messages"].append({"role": "assistant", "content": bot_reply})
        
        # Guardar en el historial general
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user_conversations[user_id].append({
            "type": "chat",
            "user_message": user_message,
            "bot_reply": bot_reply,
            "model": model_key,
            "tokens": total_tokens,
            "timestamp": timestamp
        })
        
        # Eliminar mensaje de espera y enviar respuesta
        wait_message.delete()
        update.message.reply_text(bot_reply)
        
    except Exception as e:
        wait_message.delete()
        error_message = str(e)
        logger.error(f"Error al generar la respuesta para usuario {user_id}: {error_message}")
        
        # Mensaje de error más detallado
        if "maximum context length" in error_message.lower():
            # Reiniciar contexto si es demasiado largo
            context.user_data["messages"] = context.user_data["messages"][-2:]
            update.message.reply_text(
                "❌ La conversación es demasiado larga. He reiniciado el contexto.\n"
                "Por favor, intenta enviar tu mensaje nuevamente."
            )
        else:
            update.message.reply_text(
                f"❌ Lo siento, ocurrió un error al generar la respuesta.\n"
                f"Error: {error_message[:100]}..."
            )


# Comando para mostrar el uso actual
@track_usage
def show_usage(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    tokens_used = user_token_usage[user_id]
    percentage = (tokens_used / MAX_TOKENS_PER_USER) * 100
    
    # Obtener el modelo actual
    model_key = context.user_data.get("model", "gpt-35-turbo")
    model_info = models[model_key]
    
    # Calcular costo aproximado
    input_cost = (tokens_used / 2) * (model_info["input_cost"] / 1000)  # Asumiendo que la mitad son tokens de entrada
    output_cost = (tokens_used / 2) * (model_info["output_cost"] / 1000)  # Asumiendo que la mitad son tokens de salida
    total_cost = input_cost + output_cost
    
    update.message.reply_text(
        f"📊 *Estadísticas de uso*\n\n"
        f"*Tokens utilizados:* {tokens_used} de {MAX_TOKENS_PER_USER} ({percentage:.1f}%)\n"
        f"*Modelo actual:* {model_info['name']}\n"
        f"*Costo aproximado:* ${total_cost:.4f}\n\n"
        f"*Conversaciones guardadas:* {len(user_conversations[user_id])}\n"
        f"*Límite de conversaciones:* {MAX_CONVERSATIONS_STORED}",
        parse_mode=ParseMode.MARKDOWN
    )

# Comando para reiniciar la conversación
@track_usage
def reset_conversation(update: Update, context: CallbackContext):
    user_id = update.effective_user.id
    
    if "messages" in context.user_data:
        # Guardar la conversación antes de reiniciarla
        if context.user_data["messages"]:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            user_conversations[user_id].append({
                "type": "full_conversation",
                "messages": context.user_data["messages"].copy(),
                "timestamp": timestamp
            })
        
        # Reiniciar la conversación
        context.user_data["messages"] = []
        logger.info(f"Usuario {user_id} ha reiniciado su conversación")
        update.message.reply_text("🔄 Conversación reiniciada. ¡Empecemos de nuevo!")
    else:
        update.message.reply_text("No hay una conversación activa para reiniciar.")

# Comando de ayuda detallada
@track_usage
def help_command(update: Update, context: CallbackContext):
    update.message.reply_text(
        "*🤖 Guía de uso del Bot*\n\n"
        "*Comandos disponibles:*\n\n"
        "*/start* - Inicia el bot y muestra el mensaje de bienvenida\n"
        "*/help* - Muestra esta guía de ayuda detallada\n"
        "*/setmodel <nombre>* - Cambia el modelo de IA a utilizar\n"
        "*/generatehd <descripción>* - Genera una imagen en alta definición (1920x1080)\n"
        "*/reset* - Reinicia la conversación actual, eliminando el contexto\n"
        "*/usage* - Muestra estadísticas de tu uso actual\n\n"
        "*Consejos:*\n"
        "• Para obtener mejores respuestas, sé claro y específico en tus preguntas\n"
        "• El bot mantiene el contexto de la conversación, así que puedes hacer preguntas de seguimiento\n"
        "• Si el bot parece confundido, usa /reset para empezar una nueva conversación\n"
        "• Para generar imágenes, usa descripciones detalladas para mejores resultados\n\n"
        "*Modelos disponibles:*\n"
        "• gpt-4 - El más avanzado, mejor para tareas complejas\n"
        "• gpt-4o - Versión optimizada de GPT-4\n"
        "• gpt-35-turbo - Rápido y eficiente para la mayoría de tareas\n"
        "• dall-e-3 - Para generación de imágenes\n\n"
        "Si tienes problemas o sugerencias, contacta al administrador del bot.",
        parse_mode=ParseMode.MARKDOWN
    )

# Flask route for webhook
@app.route(f'/{TELEGRAM_BOT_TOKEN}', methods=['POST'])
def webhook():
    """Handle incoming updates from Telegram"""
    update = Update.de_json(request.get_json(force=True), bot)
    dispatcher.process_update(update)
    return 'ok'

@app.route('/')
def index():
    """Health check endpoint"""
    return 'Bot is running!'

def setup_dispatcher():
    """Setup and return the dispatcher with all handlers"""
    dispatcher = Dispatcher(bot, None, workers=0)
    
    # Register command handlers
    dispatcher.add_handler(CommandHandler("start", start))
    dispatcher.add_handler(CommandHandler("help", help_command))
    dispatcher.add_handler(CommandHandler("setmodel", set_model, pass_args=True))
    dispatcher.add_handler(CommandHandler("generatehd", generate_image, pass_args=True))
    dispatcher.add_handler(CommandHandler("reset", reset_conversation))
    dispatcher.add_handler(CommandHandler("usage", show_usage))
    dispatcher.add_handler(MessageHandler(Filters.text & ~Filters.command, chat))
    
    # Register error handler
    dispatcher.add_error_handler(lambda update, context: logger.error(f"Error in update {update}: {context.error}"))
    
    return dispatcher

def main():
    # Verify that API keys are configured
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, TELEGRAM_BOT_TOKEN]):
        logger.critical("Missing required environment variables. Please configure the .env file")
        print("ERROR: Missing required environment variables. Please configure the .env file")
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
            
            # Set webhook
            bot.set_webhook(url=f"{WEBHOOK_URL}/{TELEGRAM_BOT_TOKEN}")
            
            # Start Flask server
            logger.info("Bot started successfully in webhook mode")
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
