# ============================================
# PROJET MINI IA - PRÉDICTION BTCUSDT (UP/DOWN)
# ============================================

# ----------- 1. Importation des librairies -----------

import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import ta

# ----------- 2. Connexion à Binance -----------

# Binance permet l'accès public aux données historiques
client = Client()

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1DAY

# Récupération de 3 ans de données journalières
klines = client.get_historical_klines(symbol, interval, "3 years ago UTC")

# ----------- 3. Création du DataFrame -----------

columns = [
    "timestamp", "open", "high", "low", "close", "volume",
    "close_time", "quote_asset_volume", "number_of_trades",
    "taker_buy_base", "taker_buy_quote", "ignore"
]

df = pd.DataFrame(klines, columns=columns)

# Conversion des colonnes numériques
df["close"] = df["close"].astype(float)
df["open"] = df["open"].astype(float)
df["high"] = df["high"].astype(float)
df["low"] = df["low"].astype(float)
df["volume"] = df["volume"].astype(float)

# ----------- 4. Création des indicateurs techniques -----------

# Moyennes mobiles
df["ma10"] = df["close"].rolling(window=10).mean()
df["ma20"] = df["close"].rolling(window=20).mean()

# RSI
df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()

# MACD
macd = ta.trend.MACD(df["close"])
df["macd"] = macd.macd()

# Rendement journalier
df["return"] = df["close"].pct_change()

# ----------- 5. Création de la variable cible (Target) -----------

# 1 si le prix de demain est supérieur à aujourd’hui, sinon 0
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# Suppression des valeurs manquantes
df.dropna(inplace=True)

# ----------- 6. Sélection des variables explicatives -----------

features = ["ma10", "ma20", "rsi", "macd", "volume", "return"]

X = df[features]
y = df["target"]

# ----------- 7. Séparation train / test -----------

# Important: shuffle=False car données temporelles
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ----------- 8. Entraînement du modèle -----------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------- 9. Évaluation du modèle -----------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy du modèle :", accuracy)
print("\nRapport de classification :")
print(classification_report(y_test, predictions))

# ----------- 10. Prédiction pour demain -----------

last_data = X.iloc[-1:]
prediction = model.predict(last_data)

if prediction[0] == 1:
    print("\nPrédiction pour demain : 📈 UP (Hausse)")
else:
    print("\nPrédiction pour demain : 📉 DOWN (Baisse)")