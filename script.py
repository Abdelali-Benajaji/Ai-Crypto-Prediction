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
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# ----------- 11. Génération des signaux de trading -----------

df_test = df.iloc[-len(X_test):].copy()
df_test["prediction"] = predictions

df_test["signal"] = 0
df_test["position"] = 0

for i in range(len(df_test)):
    if i == 0:
        if df_test["prediction"].iloc[i] == 1:
            df_test.loc[df_test.index[i], "signal"] = 1
            df_test.loc[df_test.index[i], "position"] = 1
    else:
        prev_position = df_test["position"].iloc[i-1]
        current_prediction = df_test["prediction"].iloc[i]
        
        if prev_position == 0 and current_prediction == 1:
            df_test.loc[df_test.index[i], "signal"] = 1
            df_test.loc[df_test.index[i], "position"] = 1
        elif prev_position == 1 and current_prediction == 0:
            df_test.loc[df_test.index[i], "signal"] = -1
            df_test.loc[df_test.index[i], "position"] = 0
        else:
            df_test.loc[df_test.index[i], "position"] = prev_position

# ----------- 12. Visualisation graphique interactive -----------

print("\nGénération des graphiques interactifs...")

buy_signals = df_test[df_test["signal"] == 1]
sell_signals = df_test[df_test["signal"] == -1]

# Création d'une figure avec plusieurs sous-graphiques
fig = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=('Prix BTC-USD avec Signaux de Trading', 'Volume', 'RSI', 'MACD'),
    row_heights=[0.5, 0.15, 0.15, 0.2]
)

# 1. Graphique principal - Prix avec signaux
fig.add_trace(
    go.Candlestick(
        x=df_test.index,
        open=df_test['open'],
        high=df_test['high'],
        low=df_test['low'],
        close=df_test['close'],
        name='Prix',
        increasing_line_color='#00ff00',
        decreasing_line_color='#ff0000'
    ),
    row=1, col=1
)

# Moyennes mobiles
fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test['ma10'],
        name='MA 10',
        line=dict(color='#00d4ff', width=1.5),
        opacity=0.7
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test['ma20'],
        name='MA 20',
        line=dict(color='#ff00ff', width=1.5),
        opacity=0.7
    ),
    row=1, col=1
)

# Signaux d'achat
fig.add_trace(
    go.Scatter(
        x=buy_signals.index,
        y=buy_signals['close'],
        mode='markers',
        name='OPEN (Achat)',
        marker=dict(
            symbol='triangle-up',
            size=15,
            color='#00ff00',
            line=dict(color='white', width=2)
        )
    ),
    row=1, col=1
)

# Signaux de vente
fig.add_trace(
    go.Scatter(
        x=sell_signals.index,
        y=sell_signals['close'],
        mode='markers',
        name='CLOSE (Vente)',
        marker=dict(
            symbol='triangle-down',
            size=15,
            color='#ff0000',
            line=dict(color='white', width=2)
        )
    ),
    row=1, col=1
)

# 2. Volume
colors = ['#00ff00' if row['close'] >= row['open'] else '#ff0000' for idx, row in df_test.iterrows()]
fig.add_trace(
    go.Bar(
        x=df_test.index,
        y=df_test['volume'],
        name='Volume',
        marker_color=colors,
        opacity=0.5,
        showlegend=False
    ),
    row=2, col=1
)

# 3. RSI
fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test['rsi'],
        name='RSI',
        line=dict(color='#ffaa00', width=2),
        showlegend=False
    ),
    row=3, col=1
)

# Lignes de surachat/survente RSI
fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)

# 4. MACD
fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test['macd'],
        name='MACD',
        line=dict(color='#00ffff', width=2),
        showlegend=False
    ),
    row=4, col=1
)

fig.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.3, row=4, col=1)

# Configuration du layout avec fond noir
fig.update_layout(
    title={
        'text': 'Signaux de Trading BTC-USD - Prédictions IA',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 24, 'color': 'white'}
    },
    template='plotly_dark',
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(color='white'),
    height=1200,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        bgcolor='rgba(0,0,0,0.5)',
        bordercolor='white',
        borderwidth=1
    ),
    hovermode='x unified',
    xaxis4=dict(
        rangeslider=dict(visible=False),
        type='date',
        gridcolor='#333333'
    )
)

# Mise à jour des axes Y
fig.update_yaxes(title_text="Prix (USD)", row=1, col=1, gridcolor='#333333')
fig.update_yaxes(title_text="Volume", row=2, col=1, gridcolor='#333333')
fig.update_yaxes(title_text="RSI", row=3, col=1, gridcolor='#333333', range=[0, 100])
fig.update_yaxes(title_text="MACD", row=4, col=1, gridcolor='#333333')

# Mise à jour de tous les axes X
for i in range(1, 5):
    fig.update_xaxes(gridcolor='#333333', row=i, col=1)

# Sauvegarde du graphique HTML
output_file = "trading_dashboard.html"
fig.write_html(output_file)
print(f"Graphique interactif sauvegardé : {output_file}")

# Affichage dans le navigateur web
fig.show()
print("\nGraphique interactif ouvert dans votre navigateur web!")
