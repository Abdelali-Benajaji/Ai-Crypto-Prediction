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
from sklearn.preprocessing import StandardScaler
import ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------- 2. Téléchargement des données -----------

print("Téléchargement des données historiques...")

ticker = yf.Ticker("BTC-USD")

df = ticker.history(period="3y", interval="1d")

print(f"Données récupérées: {len(df)} jours")

# ----------- 3. Préparation du DataFrame -----------

df.columns = df.columns.str.lower()

# ----------- 4. Création des indicateurs techniques -----------

# Moyennes mobiles multiples
df["ma5"] = df["close"].rolling(window=5).mean()
df["ma10"] = df["close"].rolling(window=10).mean()
df["ma20"] = df["close"].rolling(window=20).mean()
df["ma50"] = df["close"].rolling(window=50).mean()
df["ma100"] = df["close"].rolling(window=100).mean()

# Écart par rapport aux moyennes mobiles
df["price_to_ma10"] = df["close"] / df["ma10"]
df["price_to_ma20"] = df["close"] / df["ma20"]
df["price_to_ma50"] = df["close"] / df["ma50"]

# Croisements de moyennes mobiles
df["ma10_ma20_cross"] = df["ma10"] - df["ma20"]
df["ma20_ma50_cross"] = df["ma20"] - df["ma50"]

# RSI
df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
df["rsi_7"] = ta.momentum.RSIIndicator(df["close"], window=7).rsi()
df["rsi_21"] = ta.momentum.RSIIndicator(df["close"], window=21).rsi()

# MACD
macd = ta.trend.MACD(df["close"])
df["macd"] = macd.macd()
df["macd_signal"] = macd.macd_signal()
df["macd_diff"] = macd.macd_diff()

# Bandes de Bollinger
bollinger = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
df["bb_upper"] = bollinger.bollinger_hband()
df["bb_lower"] = bollinger.bollinger_lband()
df["bb_middle"] = bollinger.bollinger_mavg()
df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

# Stochastic Oscillator
stoch = ta.momentum.StochasticOscillator(df["high"], df["low"], df["close"], window=14, smooth_window=3)
df["stoch_k"] = stoch.stoch()
df["stoch_d"] = stoch.stoch_signal()

# ATR (Average True Range) - Volatilité
df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
df["atr_ratio"] = df["atr"] / df["close"]

# Momentum et rendements
df["return_1d"] = df["close"].pct_change(1)
df["return_3d"] = df["close"].pct_change(3)
df["return_7d"] = df["close"].pct_change(7)
df["return_14d"] = df["close"].pct_change(14)
df["return_30d"] = df["close"].pct_change(30)

# Volatilité historique
df["volatility_7d"] = df["return_1d"].rolling(window=7).std()
df["volatility_30d"] = df["return_1d"].rolling(window=30).std()

# Volume indicators
df["volume_ma10"] = df["volume"].rolling(window=10).mean()
df["volume_ratio"] = df["volume"] / df["volume_ma10"]
df["volume_change"] = df["volume"].pct_change()

# OBV (On-Balance Volume)
df["obv"] = ta.volume.OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume()
df["obv_ma10"] = df["obv"].rolling(window=10).mean()

# ADX (Average Directional Index) - Force de la tendance
df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"], window=14).adx()

# CCI (Commodity Channel Index)
df["cci"] = ta.trend.CCIIndicator(df["high"], df["low"], df["close"], window=20).cci()

# Williams %R
df["williams_r"] = ta.momentum.WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=14).williams_r()

# Prix min/max sur différentes périodes
df["high_7d"] = df["high"].rolling(window=7).max()
df["low_7d"] = df["low"].rolling(window=7).min()
df["high_30d"] = df["high"].rolling(window=30).max()
df["low_30d"] = df["low"].rolling(window=30).min()
df["position_7d"] = (df["close"] - df["low_7d"]) / (df["high_7d"] - df["low_7d"])
df["position_30d"] = (df["close"] - df["low_30d"]) / (df["high_30d"] - df["low_30d"])

# Lag features (valeurs décalées)
df["close_lag1"] = df["close"].shift(1)
df["close_lag2"] = df["close"].shift(2)
df["close_lag3"] = df["close"].shift(3)
df["return_lag1"] = df["return_1d"].shift(1)
df["return_lag2"] = df["return_1d"].shift(2)
df["volume_lag1"] = df["volume"].shift(1)

# ----------- 5. Création de la variable cible (Target) -----------

df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)

# Suppression des valeurs manquantes
df.dropna(inplace=True)

print(f"Données après nettoyage: {len(df)} jours")

# ----------- 6. Sélection des variables explicatives -----------

features = [
    # Moyennes mobiles
    "ma5", "ma10", "ma20", "ma50", "ma100",
    "price_to_ma10", "price_to_ma20", "price_to_ma50",
    "ma10_ma20_cross", "ma20_ma50_cross",
    
    # RSI
    "rsi", "rsi_7", "rsi_21",
    
    # MACD
    "macd", "macd_signal", "macd_diff",
    
    # Bollinger Bands
    "bb_width", "bb_position",
    
    # Stochastic
    "stoch_k", "stoch_d",
    
    # Volatilité
    "atr", "atr_ratio", "volatility_7d", "volatility_30d",
    
    # Momentum et rendements
    "return_1d", "return_3d", "return_7d", "return_14d", "return_30d",
    
    # Volume
    "volume", "volume_ratio", "volume_change", "obv", "obv_ma10",
    
    # Autres indicateurs
    "adx", "cci", "williams_r",
    
    # Position dans les ranges
    "position_7d", "position_30d",
    
    # Lag features
    "close_lag1", "close_lag2", "close_lag3",
    "return_lag1", "return_lag2", "volume_lag1"
]

X = df[features]
y = df["target"]

# ----------- 7. Normalisation des features -----------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features, index=X.index)

# ----------- 8. Séparation train / test -----------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

print(f"\nDonnées d'entraînement: {len(X_train)} jours")
print(f"Données de test: {len(X_test)} jours")

# ----------- 9. Entraînement du modèle -----------

print("\nEntraînement du modèle...")

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ----------- 10. Évaluation du modèle -----------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n" + "="*50)
print("RÉSULTATS DU MODÈLE")
print("="*50)
print(f"Accuracy du modèle : {accuracy:.2%}")
print("\nRapport de classification :")
print(classification_report(y_test, predictions))

# Importance des features
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 des features les plus importantes:")
print(feature_importance.head(10).to_string(index=False))

# ----------- 11. Prédiction pour demain -----------

last_data = X_scaled.iloc[-1:]
prediction = model.predict(last_data)
prediction_proba = model.predict_proba(last_data)

print("\n" + "="*50)
if prediction[0] == 1:
    print(f"Prédiction pour demain : 📈 UP (Hausse) - Confiance: {prediction_proba[0][1]:.2%}")
else:
    print(f"Prédiction pour demain : 📉 DOWN (Baisse) - Confiance: {prediction_proba[0][0]:.2%}")
print("="*50)

# ----------- 12. Génération des signaux de trading -----------

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

# ----------- 13. Visualisation graphique interactive -----------

print("\nGénération des graphiques interactifs...")

buy_signals = df_test[df_test["signal"] == 1]
sell_signals = df_test[df_test["signal"] == -1]

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

fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test['ma50'],
        name='MA 50',
        line=dict(color='#ffaa00', width=1.5),
        opacity=0.7
    ),
    row=1, col=1
)

# Bandes de Bollinger
fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test['bb_upper'],
        name='BB Upper',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5
    ),
    row=1, col=1
)

fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test['bb_lower'],
        name='BB Lower',
        line=dict(color='gray', width=1, dash='dash'),
        opacity=0.5,
        fill='tonexty',
        fillcolor='rgba(128,128,128,0.1)'
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

fig.add_trace(
    go.Scatter(
        x=df_test.index,
        y=df_test['macd_signal'],
        name='MACD Signal',
        line=dict(color='#ff6600', width=2),
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
