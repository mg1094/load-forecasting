import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

class LSTMLoadPredictor:
    def __init__(self, sequence_length=24, n_features=12, prediction_steps=1):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.prediction_steps = prediction_steps
        self.model = None
        self.history = None
        
    def build_lstm_model(self, lstm_units=[64, 32], dropout_rate=0.2, learning_rate=0.001):
        model = Sequential()
        
        # 第一个LSTM层
        model.add(LSTM(
            units=lstm_units[0], 
            return_sequences=True if len(lstm_units) > 1 else False,
            input_shape=(self.sequence_length, self.n_features)
        ))
        model.add(Dropout(dropout_rate))
        
        # 额外的LSTM层
        for i, units in enumerate(lstm_units[1:]):
            model.add(LSTM(
                units=units,
                return_sequences=True if i < len(lstm_units) - 2 else False
            ))
            model.add(Dropout(dropout_rate))
        
        # 输出层
        model.add(Dense(self.prediction_steps, activation='linear'))
        
        # 编译模型
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        print("LSTM模型构建完成!")
        print(model.summary())
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              patience=10, save_path='models/lstm_model.h5'):
        if self.model is None:
            raise ValueError("请先构建模型!")
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1),
            ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]
        
        print("开始训练...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        print("训练完成!")
        return self.history
    
    def predict(self, X):
        if self.model is None:
            raise ValueError("模型未训练或加载!")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test, preprocessor=None):
        predictions = self.predict(X_test)
        
        if preprocessor is not None:
            y_test_original = preprocessor.inverse_transform_load(y_test.flatten())
            predictions_original = preprocessor.inverse_transform_load(predictions.flatten())
        else:
            y_test_original = y_test.flatten()
            predictions_original = predictions.flatten()
        
        mae = mean_absolute_error(y_test_original, predictions_original)
        mse = mean_squared_error(y_test_original, predictions_original)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_original, predictions_original)
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        
        metrics = {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R²': r2, 'MAPE': mape}
        
        print("=== 模型评估结果 ===")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        return metrics, predictions_original, y_test_original
    
    def plot_training_history(self):
        if self.history is None:
            print("没有训练历史可绘制!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(self.history.history['mae'], label='Training MAE')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='models/lstm_model.h5'):
        if self.model is None:
            raise ValueError("没有模型可保存!")
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath='models/lstm_model.h5'):
        self.model = tf.keras.models.load_model(filepath, compile=False)
        print(f"模型已从 {filepath} 加载")
        return self.model