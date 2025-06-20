# This code implements a simple web server using aiohttp that serves a webpage for hand gesture recognition using a TensorFlow model.
# The server handles image requests, serves an HTML page, and processes WebSocket connections for real-time hand gesture recognition.

# First, ensure you have the required packages installed:
import os 
import asyncio
import aiohttp  
from aiohttp import web, WSMsgType  
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.optimizers            import SGD
from tensorflow.keras                       import regularizers
from tensorflow.keras.models                import Sequential, Model
from tensorflow.keras.layers                import *
from tensorflow.keras.models import load_model
import keras
import mediapipe as mp
import ssl


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage for this script


# Initialize MediaPipe Hands for hand detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_mp = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Define constants
HTML_DIR = "html" # Directory where HTML files are stored
model_path = 'best_model_key.keras' # Path to the trained TensorFlow model
image_side = 512 # Size of the input image for the model
# Define the classes corresponding to the model's output
classes = ("A", "B", "C", "CH", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "W", "X")

# Load the TensorFlow model
model = tf.keras.models.load_model(model_path)
model_mutex = asyncio.Lock() 

# Ensure the model is compiled
async def img_handler(request):
    img = request.match_info['img'].split(".")[0]  
    images_folder = os.path.join(HTML_DIR, "example_imgs")
    file_path = os.path.join(images_folder, f"{img}.webp")

    if img in classes and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            content = f.read()
            return web.Response(body=content, content_type='image/webp')
    return web.Response(text='File not found', status=404)

# This function serves the main HTML page
# It reads the index.html file from the HTML_DIR and returns it as a response
async def http_handler(request):
    index_path = os.path.join(HTML_DIR, "index.html")
    try:
        with open(index_path, "rb") as f:
            content = f.read()
            return web.Response(body=content, content_type='text/html')
    except FileNotFoundError:
        return web.Response(text='File not found', status=404)

# This function handles WebSocket connections for real-time hand gesture recognition
# It processes incoming messages, performs hand detection, and sends back predictions
async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)

    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT and msg.data == 'close':
            await ws.close()
        elif msg.type == aiohttp.WSMsgType.BINARY:
            frame = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
            # frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_mp.process(rgb)

            black = np.zeros_like(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(black, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    # mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                processed = cv2.resize(black, (image_side, image_side))
                # processed = cv2.resize(frame, (image_side, image_side))
                input_image = processed.astype(np.float32) / 255.0
                input_image = np.expand_dims(input_image, axis=0)

                async with model_mutex:
                    pred = model.predict(input_image)
                    index = np.argmax(pred)
                    percent = round(pred[0][index] * 100, 2)

                    if percent < 20:
                        await ws.send_str("🤷‍♂️ No prediction")
                    else:
                        await ws.send_str(f"{classes[index]} ({percent}%)")
            else:
                await ws.send_str("✋ No hand detected")
    return ws

# This function creates the web application and sets up the routes
# It maps the URL paths to the corresponding handler functions
def create_app():
    app = web.Application()
    app.add_routes([
        web.get('/', http_handler),
        web.get('/ws', websocket_handler),
        web.get('/example_imgs/{img}', img_handler),
        web.get('/{path:.*}', http_handler)
    ])
    return app

# This function starts the web server
# It sets up the SSL context for secure connections and starts the server on the specified port
async def start_server():
    # Use the PORT environment variable from Heroku if available
    port = int(os.environ.get("PORT", 8182))
    host = '0.0.0.0'
    
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain('server.crt', 'server.key')
    
    app = create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port, ssl_context=ssl_context)
    await site.start()

    print(f"Server running on https://{host}:{port}")
    await asyncio.Event().wait()

if __name__ == "__main__":
    asyncio.run(start_server())


# To run this script, you need to run in two different terminals the next commands:
# 1. Start the ngrok tunnel to expose your local server to the internet:
# ./ngrok.exe http https://127.0.0.1:8182 --url=ultimate-quagga-presumably.ngrok-free.app
# 2. Start the web server:
# python implementacion_web.py
