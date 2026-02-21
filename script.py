# ============================================
# PROJET MINI IA - PRÉDICTION BTCUSDT (UP/DOWN)
# ============================================

# ----------- 1. Importation des librairies -----------

import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import ta

# ----------- 2. Téléchargement des données -----------

# Utilisation de yfinance pour récupérer les données BTC-USD
# (alternative à Binance qui fonctionne sans restrictions géographiques)
print("Téléchargement des données historiques...")

# BTC-USD est l'équivalent de BTCUSDT sur Yahoo Finance
ticker = yf.Ticker("BTC-USD")

# Récupération de 3 ans de données journalières
df = ticker.history(period="3y", interval="1d")

print(f"Données récupérées: {len(df)} jours")

# ----------- 3. Préparation du DataFrame -----------

# Renommage des colonnes en minuscules pour cohérence
df.columns = df.columns.str.lower()

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

# 1 si le prix de demain est supérieur à aujourd'hui, sinon 0
df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# Suppression des valeurs manquantes
df.dropna(inplace=True)

print(f"Données après nettoyage: {len(df)} jours")

# ----------- 6. Sélection des variables explicatives -----------

features = ["ma10", "ma20", "rsi", "macd", "volume", "return"]

X = df[features]
y = df["target"]

# ----------- 7. Séparation train / test -----------

# Important: shuffle=False car données temporelles
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print(f"\nDonnées d'entraînement: {len(X_train)} jours")
print(f"Données de test: {len(X_test)} jours")

# ----------- 8. Entraînement du modèle -----------

print("\nEntraînement du modèle...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ----------- 9. Évaluation du modèle -----------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print("RÉSULTATS DU MODÈLE")
print("="*50)
print(f"Accuracy du modèle : {accuracy:.2%}")
print("\nRapport de classification :")
print(classification_report(y_test, predictions))

# ----------- 10. Prédiction pour demain -----------

last_data = X.iloc[-1:]
prediction = model.predict(last_data)

print("="*50)
if prediction[0] == 1:
    print("Prédiction pour demain : 📈 UP (Hausse)")
else:
    print("Prédiction pour demain : 📉 DOWN (Baisse)")
print("="*50)
