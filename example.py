import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------
# 1. 예제 데이터 생성
# ---------------------
# 정상 거래 1000건, 사기 거래 50건 (라벨 포함)
np.random.seed(42)
normal_data = np.random.normal(0, 1, size=(1000, 6))
fraud_data = np.random.normal(2, 1, size=(50, 6))

X = np.vstack([normal_data, fraud_data])
y = np.hstack([np.zeros(1000), np.ones(50)])

df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['label'] = y

# ---------------------
# 2. Autoencoder 훈련 (정상 거래만)
# ---------------------
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(df[df['label'] == 0].drop('label', axis=1))

input_dim = X_scaled.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(4, activation='relu')(input_layer)
latent = Dense(2, activation='relu')(encoded)
decoded = Dense(4, activation='relu')(latent)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_scaled, X_scaled, epochs=30, batch_size=32, verbose=0)

# ---------------------
# 3. 재구성 에러 계산 (전체 데이터)
# ---------------------
X_all = df.drop('label', axis=1)
X_all_scaled = scaler.transform(X_all)
X_reconstructed = autoencoder.predict(X_all_scaled, verbose=0)
reconstruction_error = np.mean(np.square(X_all_scaled - X_reconstructed), axis=1)

df['anomaly_score'] = reconstruction_error

# ---------------------
# 4. XGBoost 학습 (Autoencoder score 포함)
# ---------------------
features_for_xgb = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(features_for_xgb, df['label'], test_size=0.2, random_state=42)

xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, y_train)

# ---------------------
# 5. SHAP으로 피처 중요도 해석
# ---------------------
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

shap.summary_plot(shap_values, X_test)
