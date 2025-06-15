import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32], num_layers=2, dropout_rate=0.2, prediction_steps=1):
        """
        PyTorch LSTM模型
        
        Args:
            input_size: 输入特征数量
            hidden_sizes: 隐藏层大小列表
            num_layers: LSTM层数
            dropout_rate: Dropout比率
            prediction_steps: 预测步数
        """
        super(LSTMModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_layers = num_layers
        self.prediction_steps = prediction_steps
        
        # 第一个LSTM层
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_sizes[0],
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # 第二个LSTM层
        if len(hidden_sizes) > 1:
            self.lstm2 = nn.LSTM(
                input_size=hidden_sizes[0],
                hidden_size=hidden_sizes[1],
                num_layers=1,
                batch_first=True,
                dropout=0
            )
            self.dropout2 = nn.Dropout(dropout_rate)
            final_hidden_size = hidden_sizes[1]
        else:
            self.lstm2 = None
            final_hidden_size = hidden_sizes[0]
        
        # 输出层
        self.fc = nn.Linear(final_hidden_size, prediction_steps)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        
        # 第一个LSTM层
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)
        
        # 第二个LSTM层 (如果存在)
        if self.lstm2 is not None:
            lstm2_out, _ = self.lstm2(lstm1_out)
            lstm2_out = self.dropout2(lstm2_out)
            # 只取最后一个时间步的输出
            final_out = lstm2_out[:, -1, :]
        else:
            # 只取最后一个时间步的输出
            final_out = lstm1_out[:, -1, :]
        
        # 全连接层
        output = self.fc(final_out)
        
        return output

class LSTMLoadPredictor:
    def __init__(self, sequence_length=24, n_features=12, prediction_steps=1):
        """
        PyTorch版本的LSTM负荷预测器
        
        Args:
            sequence_length: 输入序列长度
            n_features: 特征数量
            prediction_steps: 预测步数
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.prediction_steps = prediction_steps
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.history = {'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
        
        print(f"使用设备: {self.device}")
        
    def build_model(self, hidden_sizes=[64, 32], dropout_rate=0.2, learning_rate=0.001):
        """
        构建LSTM模型
        
        Args:
            hidden_sizes: LSTM隐藏层大小列表
            dropout_rate: Dropout比率
            learning_rate: 学习率
        """
        self.model = LSTMModel(
            input_size=self.n_features,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            prediction_steps=self.prediction_steps
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 损失函数
        self.criterion = nn.MSELoss()
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        print("PyTorch LSTM模型构建完成!")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"可训练参数: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, 
              patience=15, save_path='models/lstm_model_pytorch.pth'):
        """
        训练模型
        """
        if self.model is None:
            raise ValueError("请先构建模型!")
        
        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val).to(self.device)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print("开始训练...")
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")
        print(f"批次大小: {batch_size}")
        print(f"设备: {self.device}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # 创建保存目录
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0
            train_mae = 0.0
            
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for batch_X, batch_y in train_pbar:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                train_mae += torch.mean(torch.abs(outputs - batch_y)).item()
                
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.6f}',
                    'MAE': f'{torch.mean(torch.abs(outputs - batch_y)).item():.6f}'
                })
            
            # 验证阶段
            self.model.eval()
            val_loss = 0.0
            val_mae = 0.0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]')
                for batch_X, batch_y in val_pbar:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    
                    val_loss += loss.item()
                    val_mae += torch.mean(torch.abs(outputs - batch_y)).item()
                    
                    val_pbar.set_postfix({
                        'Loss': f'{loss.item():.6f}',
                        'MAE': f'{torch.mean(torch.abs(outputs - batch_y)).item():.6f}'
                    })
            
            # 计算平均损失
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_mae = train_mae / len(train_loader)
            avg_val_mae = val_mae / len(val_loader)
            
            # 保存历史
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['train_mae'].append(avg_train_mae)
            self.history['val_mae'].append(avg_val_mae)
            
            # 学习率调度
            self.scheduler.step(avg_val_loss)
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {avg_train_loss:.6f}, Train MAE: {avg_train_mae:.6f}')
            print(f'  Val Loss: {avg_val_loss:.6f}, Val MAE: {avg_val_mae:.6f}')
            print(f'  LR: {self.optimizer.param_groups[0]["lr"]:.8f}')
            
            # 早停机制
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # 保存最佳模型
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'epoch': epoch,
                    'loss': best_val_loss,
                    'model_config': {
                        'sequence_length': self.sequence_length,
                        'n_features': self.n_features,
                        'prediction_steps': self.prediction_steps
                    }
                }, save_path)
                print(f'  ✅ 模型已保存 (Val Loss: {best_val_loss:.6f})')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f'\n早停触发! {patience} 轮验证损失未改善')
                break
            
            print('-' * 60)
        
        print(f"\n训练完成! 最佳验证损失: {best_val_loss:.6f}")
        return self.history
    
    def predict(self, X):
        """
        进行预测
        """
        if self.model is None:
            raise ValueError("模型未训练或加载!")
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(X_tensor)
            
        return predictions.cpu().numpy()
    
    def evaluate(self, X_test, y_test, preprocessor=None):
        """
        评估模型性能
        """
        predictions = self.predict(X_test)
        
        # 如果提供了预处理器，进行反标准化
        if preprocessor is not None:
            predictions_original = preprocessor.inverse_transform_load(predictions.flatten())
            y_test_original = preprocessor.inverse_transform_load(y_test.flatten())
        else:
            predictions_original = predictions.flatten()
            y_test_original = y_test.flatten()
        
        # 计算评估指标
        rmse = np.sqrt(mean_squared_error(y_test_original, predictions_original))
        mae = mean_absolute_error(y_test_original, predictions_original)
        r2 = r2_score(y_test_original, predictions_original)
        
        # 计算MAPE
        mape = np.mean(np.abs((y_test_original - predictions_original) / y_test_original)) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
        
        print("模型评估结果:")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
        return metrics, predictions_original, y_test_original
    
    def plot_training_history(self):
        """
        绘制训练历史
        """
        if not self.history['train_loss']:
            print("没有训练历史可绘制!")
            return
        
        plt.figure(figsize=(15, 5))
        
        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='训练损失', linewidth=2)
        plt.plot(self.history['val_loss'], label='验证损失', linewidth=2)
        plt.title('训练和验证损失', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_mae'], label='训练MAE', linewidth=2)
        plt.plot(self.history['val_mae'], label='验证MAE', linewidth=2)
        plt.title('训练和验证MAE', fontsize=14, fontweight='bold')
        plt.xlabel('轮次')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('models/training_history_pytorch.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath='models/lstm_model_pytorch.pth'):
        """
        保存模型
        """
        if self.model is None:
            raise ValueError("没有模型可保存!")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else None,
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'history': self.history,
            'model_config': {
                'sequence_length': self.sequence_length,
                'n_features': self.n_features,
                'prediction_steps': self.prediction_steps
            }
        }, filepath)
        
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath='models/lstm_model_pytorch.pth', hidden_sizes=[64, 32], dropout_rate=0.2):
        """
        加载模型
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 从检查点恢复配置
        if 'model_config' in checkpoint:
            config = checkpoint['model_config']
            self.sequence_length = config['sequence_length']
            self.n_features = config['n_features']
            self.prediction_steps = config['prediction_steps']
        
        # 重建模型
        self.model = LSTMModel(
            input_size=self.n_features,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate,
            prediction_steps=self.prediction_steps
        ).to(self.device)
        
        # 加载模型状态
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载训练历史
        if 'history' in checkpoint:
            self.history = checkpoint['history']
        
        print(f"模型已从 {filepath} 加载")
        print(f"模型配置: 序列长度={self.sequence_length}, 特征数={self.n_features}")
        
        return self.model