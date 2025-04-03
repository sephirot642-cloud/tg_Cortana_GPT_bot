# Bot de Telegram con ChatGPT y DALL-E

Este bot de Telegram integra las capacidades de ChatGPT y DALL-E para proporcionar respuestas de texto e imágenes generadas por IA.

## Características

- 💬 **Chat con memoria**: Mantiene el contexto de la conversación para respuestas más coherentes
- 🖼️ **Generación de imágenes**: Crea imágenes en alta definición con DALL-E 3
- 🔄 **Múltiples modelos**: Soporte para varios modelos de OpenAI (GPT-4, GPT-3.5, etc.)
- 📊 **Control de uso**: Seguimiento del uso de tokens por usuario
- 🔒 **Límites configurables**: Establece límites de uso para controlar costos
- 📝 **Logging detallado**: Registro completo de actividades para monitoreo
- 🔁 **Reintentos automáticos**: Manejo de errores con reintentos para mayor robustez

## Requisitos

- Python 3.7 o superior
- Una cuenta en Azure OpenAI o OpenAI
- Un token de bot de Telegram (obtenido a través de [@BotFather](https://t.me/BotFather))

## Instalación

### Instalación Local

1. Clona este repositorio o descarga los archivos

2. Instala las dependencias:

```bash
pip install -r requirements.txt
```

3. Configura el archivo `.env` con tus credenciales:

```
AZURE_OPENAI_API_KEY=tu_clave_api_aqui
AZURE_OPENAI_ENDPOINT=tu_endpoint_aqui
TELEGRAM_BOT_TOKEN=tu_token_de_telegram_aqui

# Configuración opcional
MAX_TOKENS_PER_USER=1000
MAX_CONVERSATIONS_STORED=10
LOG_LEVEL=INFO
```

### Despliegue en Render.com

1. Crea una cuenta en [Render.com](https://render.com) si aún no tienes una

2. Haz clic en "New" y selecciona "Web Service"

3. Conecta tu repositorio de GitHub o sube los archivos manualmente

4. Configura el servicio con los siguientes parámetros:
   - **Name**: Nombre de tu bot
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn bot:app`

5. En la sección "Environment Variables", añade las siguientes variables:

```
AZURE_OPENAI_API_KEY=tu_clave_api_aqui
AZURE_OPENAI_ENDPOINT=tu_endpoint_aqui
TELEGRAM_BOT_TOKEN=tu_token_de_telegram_aqui
RENDER=true
WEBHOOK_URL=https://tu-app-name.onrender.com
```

6. Haz clic en "Create Web Service"

7. Una vez desplegado, configura el webhook de Telegram visitando:
   `https://api.telegram.org/bot<TELEGRAM_BOT_TOKEN>/setWebhook?url=https://tu-app-name.onrender.com/<TELEGRAM_BOT_TOKEN>`

## Uso

1. Inicia el bot:

```bash
python bot.py
```

2. Abre Telegram y busca tu bot por su nombre de usuario

3. Inicia una conversación con el comando `/start`

## Comandos disponibles

- `/start` - Inicia el bot y muestra el mensaje de bienvenida
- `/help` - Muestra una guía de ayuda detallada
- `/setmodel <nombre>` - Cambia el modelo de IA a utilizar
- `/generatehd <descripción>` - Genera una imagen en alta definición (1920x1080)
- `/reset` - Reinicia la conversación actual, eliminando el contexto
- `/usage` - Muestra estadísticas de tu uso actual

## Personalización

Puedes modificar los siguientes parámetros en el archivo `.env`:

- `MAX_TOKENS_PER_USER`: Límite de tokens por usuario
- `MAX_CONVERSATIONS_STORED`: Número máximo de conversaciones almacenadas
- `LOG_LEVEL`: Nivel de detalle de los logs (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Solución de problemas

- Si el bot no responde, verifica que las credenciales en el archivo `.env` sean correctas
- Revisa los logs en `bot.log` para identificar posibles errores
- Asegúrate de tener suficientes créditos en tu cuenta de OpenAI
- Para problemas en Render.com, verifica los logs del servicio en el dashboard de Render
- Si el webhook no funciona, asegúrate de que la URL sea accesible públicamente

## Licencia

Este proyecto está disponible bajo la licencia MIT.