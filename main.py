--- IMPORTS ---

import websocket import threading import json import time import numpy as np import pickle import os import random import requests import tensorflow as tf from sklearn.ensemble import RandomForestClassifier from xgboost import XGBClassifier from sklearn.linear_model import LogisticRegression from sklearn.preprocessing import StandardScaler from sklearn.neighbors import KNeighborsClassifier from statsmodels.tsa.arima.model import ARIMA from tensorflow.keras.models import Sequential from tensorflow.keras.layers import LSTM, Dense from tensorflow.keras.optimizers import Adam from google.colab import drive import warnings import datetime

--- CONFIG TELEGRAM ---

TELEGRAM_TOKEN = "7028303454:AAE_4bzZ14gXUbEFtJAJ5iOGaXw_ucwrXgE" TELEGRAM_CHAT_ID = "7919382267"

def send_telegram(msg): try: url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage" data = {"chat_id": TELEGRAM_CHAT_ID, "text": msg} requests.post(url, data=data) except Exception as e: print(f"Erreur Telegram: {e}")

--- INITIALISATION ---

warnings.filterwarnings("ignore") tf.get_logger().setLevel('ERROR') drive.mount('/content/drive', force_remount=True)

history, X_full, y_full = [], [], [] scaler = StandardScaler() model_rf = RandomForestClassifier(n_estimators=100) model_xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss') model_meta = LogisticRegression() model_knn = KNeighborsClassifier(n_neighbors=5) model_arima = None model_lstm, retrain_interval, last_train_len = None, 1, 0

--- STATISTIQUES ---

prediction_count = 0 correct_count = 0 last_report_time = time.time()

--- MODELE LSTM ---

def build_lstm_model(input_shape): model = Sequential() model.add(LSTM(64, input_shape=input_shape, return_sequences=True)) model.add(LSTM(32)) model.add(Dense(1, activation='sigmoid')) model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy']) return model

--- FEATURES ---

def extract_features(seq): return [ np.mean(seq), np.std(seq), max(seq), min(seq), seq[-1] - seq[0], np.median(seq), np.percentile(seq, 75) - np.percentile(seq, 25) ]

def prepare_data(hist): X, y = [], [] for i in range(len(hist) - 5): window = hist[i:i+5] target = int(hist[i+5] > 2.0) X.append(extract_features(window)) y.append(target) return np.array(X), np.array(y)

--- ENTRAINEMENT ---

def train_models(): global model_lstm, X_full, y_full, scaler, model_arima X, y = np.array(X_full), np.array(y_full) if len(set(y)) < 2: return scaler.fit(X) X_scaled = scaler.transform(X) model_rf.fit(X_scaled, y) model_xgb.fit(X_scaled, y) model_knn.fit(X_scaled, y) meta_X = np.column_stack([ model_rf.predict_proba(X_scaled)[:,1], model_xgb.predict_proba(X_scaled)[:,1], model_knn.predict_proba(X_scaled)[:,1] ]) model_meta.fit(meta_X, y)

X_seq = np.array([history[i:i+5] for i in range(len(history)-5)])
y_seq = np.array([int(history[i+5] > 2.0) for i in range(len(history)-5)])
model_lstm = build_lstm_model((5, 1))
model_lstm.fit(X_seq.reshape((X_seq.shape[0], 5, 1)), y_seq, epochs=10, batch_size=8, verbose=0)

model_arima = ARIMA(history, order=(5, 1, 0))
model_arima = model_arima.fit()

--- PREDICTION ---

def predict_realtime(): global prediction_count, correct_count if len(history) < 10: return "Pas assez de données" seq = history[-5:] features = scaler.transform([extract_features(seq)])

rf = model_rf.predict_proba(features)[0][1]
xgb = model_xgb.predict_proba(features)[0][1]
knn = model_knn.predict_proba(features)[0][1]
meta = model_meta.predict_proba([[rf, xgb, knn]])[0][1]
lstm = model_lstm.predict(np.array(seq).reshape((1, 5, 1)), verbose=0)[0][0]
avg_pred = (meta + lstm) / 2
prediction = "OUI" if avg_pred > 0.5 else "NON"
confidence = round(avg_pred * 100, 1)

arima_pred = model_arima.forecast(steps=1)[0]
arima_conf = "Oui" if arima_pred > 2.0 else "Non"

prediction_count += 1
if (avg_pred > 0.5 and history[-1] > 2.0) or (avg_pred <= 0.5 and history[-1] <= 2.0):
    correct_count += 1

return f"Crash > 2.0 : {prediction} (Confiance: {confidence}%) | ARIMA: {arima_conf}"

--- RAPPORT HORAIRE ---

def hourly_report(): global prediction_count, correct_count, last_report_time while True: if time.time() - last_report_time >= 3600: precision = 100 * correct_count / prediction_count if prediction_count > 0 else 0 now = datetime.datetime.now().strftime("%H:%M") message = ( f"[RAPPORT HORAIRE]\nHeure: {now}\n" f"Prédictions totales: {prediction_count}\n" f"Correctes: {correct_count}\n" f"Précision: {precision:.2f}%" ) send_telegram(message) last_report_time = time.time() time.sleep(60)

--- SAUVEGARDE ---

def save_models(): path = "/content/drive/MyDrive/crash_models/" os.makedirs(path, exist_ok=True) with open(path + "scaler.pkl", "wb") as f: pickle.dump(scaler, f) with open(path + "rf.pkl", "wb") as f: pickle.dump(model_rf, f) with open(path + "xgb.pkl", "wb") as f: pickle.dump(model_xgb, f) with open(path + "meta.pkl", "wb") as f: pickle.dump(model_meta, f) with open(path + "knn.pkl", "wb") as f: pickle.dump(model_knn, f) pickle.dump(model_arima, open(path + "arima.pkl", "wb")) model_lstm.save(path + "lstm.keras")

def load_models(): global scaler, model_rf, model_xgb, model_meta, model_lstm, model_arima, model_knn path = "/content/drive/MyDrive/crash_models/" try: with open(path + "scaler.pkl", "rb") as f: scaler = pickle.load(f) with open(path + "rf.pkl", "rb") as f: model_rf = pickle.load(f) with open(path + "xgb.pkl", "rb") as f: model_xgb = pickle.load(f) with open(path + "meta.pkl", "rb") as f: model_meta = pickle.load(f) with open(path + "knn.pkl", "rb") as f: model_knn = pickle.load(f) model_arima = pickle.load(open(path + "arima.pkl", "rb")) model_lstm = tf.keras.models.load_model(path + "lstm.keras") except Exception as e: print(f"Erreur de chargement des modèles : {e}")

--- WEBSOCKET HANDLER ---

def on_message(ws, message): global history, X_full, y_full, last_train_len try: data = json.loads(message.strip('\x1e')) if data.get("type") == 1 and data.get("target") == "OnCrash": mult = float(data["arguments"][0].get("f", 0)) print(f">> Crash : x{mult}") history.append(mult) if len(history) > 10: X, y = prepare_data(history) if len(X): X_full, y_full = X.tolist(), y.tolist() if len(y_full) >= last_train_len + retrain_interval: print(">>> Réentraînement...") train_models() save_models() last_train_len = len(y_full) if model_lstm: result = predict_realtime() print(">>> Prédiction :", result) send_telegram(result) except Exception as e: print(f"Erreur message : {e}")

def on_error(ws, error): print(f"Erreur WebSocket : {error}") def on_close(ws, code, msg): print("Fermé, reconnexion...") time.sleep(5) start_ws()

def on_open(ws): def run(): ws.send(json.dumps({"protocol": "json", "version": 1}) + chr(0x1e)) time.sleep(1) ws.send(json.dumps({ "arguments": [{"activity": 30, "account": 1200134785}], "invocationId": "0", "target": "Account", "type": 1 }) + chr(0x1e)) threading.Thread(target=run).start()

--- DEMARRAGE ---

def start_ws(): try: load_models() threading.Thread(target=hourly_report, daemon=True).start() ws_url = "wss://l6k-b2jx-c.com/games-frame/sockets/crash?whence=22&fcountry=96&ref=304&gr=2057&appGuid=games-web-app-unknown&lng=fr_FR&access_token=eyJhbGciOiJFUzI1NiIsImtpZCI6IjEiLCJ0eXAiOiJKV1QifQ.eyJzdWIiOiI1MC8xMjAwMTM0Nzg1IiwicGlkIjoiMzA0IiwianRpIjoiMC8yNGQxYjcwYzM2YTQ4MzRhZGY4NTljNDE5ZjEwMWFiM2M5NTM4ZjE3MDY5MDhlNWE4YTdjYzJhY2ZhZTcyZTc1IiwiYXBwIjoiN2Q2Y2VmZmVmZDkzYzFiMl8yIiwieHBqIjoiMCIsInhnciI6IjIwNTciLCJuYmYiOjE3NDY4NzE5NjYsImV4cCI6MTc0Njg3MzE2NiwiaWF0IjoxNzQ2ODcxOTY2fQ.roeBiLLlm8ai9LY7dh-WhGdO8-G3_14KeHhKK6498e__HlomrlCqeTiBGjXVqz2e3wqw04lRR1dql8fbxBVL2g" ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error, on_close=on_close) ws.on_open = on_open ws.run_forever() except Exception as e: print(f"Erreur WebSocket: {e}") time.sleep(5) start_ws()

--- LANCEMENT FINAL ---

if name == "main": print(">>> Lancement du bot de prédiction Crash en temps réel...") start_ws()

