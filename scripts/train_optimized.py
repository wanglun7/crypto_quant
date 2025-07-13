#!/usr/bin/env python3
"""
优化版CNN-LSTM训练脚本 - 30天数据 + 严格时间序列分割
目标：超越HODL，实现爆炸收益

关键改进：
1. 使用30天完整数据（vs 3天）
2. 严格时间序列分割，防止数据泄漏
3. 扩展技术指标库
4. 规范化验证流程
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import structlog
from pathlib import Path

from crypto_quant.models.cnn_lstm import CNN_LSTM
from crypto_quant.utils.indicators import add_cnn_lstm_features, prepare_cnn_lstm_data

logger = structlog.get_logger(__name__)


class TimeSeriesSplitter:
    """严格的时间序列分割器，防止数据泄漏"""
    
    def __init__(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
    
    def split(self, df: pd.DataFrame):
        """按时间顺序分割数据，确保无未来信息泄漏"""
        n = len(df)
        
        # 计算分割点
        train_end = int(n * self.train_ratio)
        val_end = int(n * (self.train_ratio + self.val_ratio))
        
        # 严格按时间顺序分割
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        logger.info(
            "时间序列分割完成",
            train_size=len(train_df),
            val_size=len(val_df),
            test_size=len(test_df),
            train_period=f"{train_df['timestamp'].min()} ~ {train_df['timestamp'].max()}",
            val_period=f"{val_df['timestamp'].min()} ~ {val_df['timestamp'].max()}",
            test_period=f"{test_df['timestamp'].min()} ~ {test_df['timestamp'].max()}"
        )
        
        return train_df, val_df, test_df


class OptimizedTrainer:
    """优化版训练器，专注于超越HODL"""
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            
            # 收集预测和目标
            preds = (outputs > 0.5).cpu().numpy().astype(int)
            all_preds.extend(preds.flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())
        
        # 计算指标
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        return avg_loss, accuracy, f1
    
    def validate(self, val_loader, criterion):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()
                
                # 收集预测、概率和目标
                probs = outputs.cpu().numpy().flatten()
                preds = (outputs > 0.5).cpu().numpy().astype(int)
                
                all_probs.extend(probs)
                all_preds.extend(preds.flatten())
                all_targets.extend(y_batch.cpu().numpy().flatten())
        
        # 计算指标
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_preds)
        precision = precision_score(all_targets, all_preds, zero_division=0)
        recall = recall_score(all_targets, all_preds, zero_division=0)
        f1 = f1_score(all_targets, all_preds, zero_division=0)
        
        # AUC需要至少两个类别
        auc = roc_auc_score(all_targets, all_probs) if len(set(all_targets)) > 1 else 0.5
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc,
            'prob_mean': np.mean(all_probs),
            'prob_std': np.std(all_probs)
        }
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """完整训练流程"""
        
        # 优化器和损失函数
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCELoss()
        
        # 学习率调度
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_f1 = 0
        patience_counter = 0
        patience_limit = 10
        
        logger.info("开始训练", epochs=epochs, lr=lr)
        
        for epoch in range(epochs):
            # 训练
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, optimizer, criterion)
            
            # 验证
            val_metrics = self.validate(val_loader, criterion)
            
            # 学习率调度
            scheduler.step(val_metrics['loss'])
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_metrics['accuracy'])
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_metrics['f1_score'])
            
            # 早停检查
            if val_metrics['f1_score'] > best_val_f1:
                best_val_f1 = val_metrics['f1_score']
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'models/best_cnn_lstm_30d.pth')
            else:
                patience_counter += 1
            
            # 定期输出
            if epoch % 5 == 0 or epoch == epochs - 1:
                logger.info(
                    "训练进度",
                    epoch=epoch,
                    train_loss=round(train_loss, 4),
                    val_loss=round(val_metrics['loss'], 4),
                    train_acc=round(train_acc, 4),
                    val_acc=round(val_metrics['accuracy'], 4),
                    val_f1=round(val_metrics['f1_score'], 4),
                    val_auc=round(val_metrics['auc_score'], 4),
                    prob_range=f"[{val_metrics['prob_mean']:.3f}±{val_metrics['prob_std']:.3f}]"
                )
            
            # 早停
            if patience_counter >= patience_limit:
                logger.info("早停触发", epoch=epoch, best_f1=best_val_f1)
                break
        
        return self.history


def create_data_loaders(train_df, val_df, sequence_length=60, batch_size=32):
    """创建数据加载器"""
    
    # 准备训练数据
    X_train, y_train = prepare_cnn_lstm_data(train_df, sequence_length=sequence_length)
    X_val, y_val = prepare_cnn_lstm_data(val_df, sequence_length=sequence_length)
    
    logger.info(
        "数据准备完成",
        train_samples=len(X_train),
        val_samples=len(X_val),
        features=X_train.shape[2],
        sequence_length=sequence_length,
        positive_ratio_train=y_train.mean(),
        positive_ratio_val=y_val.mean()
    )
    
    # 转换为PyTorch张量
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).unsqueeze(1)
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val).unsqueeze(1)
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False  # 时间序列不shuffle
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader


def main(test_mode=False):
    """主训练流程"""
    
    # 配置日志
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    logger.info("开始30天数据优化训练")
    
    # 1. 加载30天数据
    try:
        df = pd.read_csv('data/BTC_USDT_1m_extended_30days.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')  # 确保时间顺序
        
        logger.info(
            "数据加载完成",
            rows=len(df),
            period=f"{df['timestamp'].min()} ~ {df['timestamp'].max()}",
            days=(df['timestamp'].max() - df['timestamp'].min()).days
        )
    except FileNotFoundError:
        logger.error("30天数据文件不存在，请先获取数据")
        return
    
    # 2. 添加技术指标
    df_with_features = add_cnn_lstm_features(df)
    
    # 3. 严格时间序列分割
    splitter = TimeSeriesSplitter(train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
    train_df, val_df, test_df = splitter.split(df_with_features)
    
    # 4. 创建数据加载器
    train_loader, val_loader = create_data_loaders(
        train_df, val_df, 
        sequence_length=60,  # 增加序列长度
        batch_size=32       # 较小批次以获得更稳定训练
    )
    
    # 5. 创建优化模型
    model = CNN_LSTM(
        n_features=18,
        sequence_length=60,
        cnn_filters=[32, 64, 128],  # 保持原架构
        lstm_hidden_size=64,
        lstm_num_layers=2,
        dropout_rate=0.3,           # 增加正则化
        fc_hidden_size=64
    )
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info("使用设备", device=str(device))
    
    # 6. 训练模型
    trainer = OptimizedTrainer(model, device=device)
    
    if test_mode:
        # 测试模式：快速验证流程
        epochs = 5
        logger.info("运行测试模式", epochs=epochs)
    else:
        # 生产模式：完整训练
        epochs = 100
        logger.info("运行生产模式", epochs=epochs)
    
    history = trainer.train(
        train_loader, val_loader,
        epochs=epochs,
        lr=0.0005    # 较小学习率，更稳定
    )
    
    # 7. 最终验证
    logger.info("训练完成，进行最终测试...")
    
    # 在测试集上评估
    test_loader, _ = create_data_loaders(test_df, test_df, sequence_length=60, batch_size=32)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('models/best_cnn_lstm_30d.pth'))
    trainer.model = model.to(device)
    
    test_metrics = trainer.validate(test_loader, nn.BCELoss())
    
    logger.info("最终测试结果", **test_metrics)
    
    # 8. 保存训练历史
    import json
    training_summary = {
        'data_info': {
            'total_samples': len(df),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'sequence_length': 60
        },
        'final_metrics': test_metrics,
        'training_history': {k: [float(x) for x in v] for k, v in history.items()},
        'model_params': sum(p.numel() for p in model.parameters()),
        'training_completed': datetime.now().isoformat()
    }
    
    with open('results/training_summary_30d.json', 'w') as f:
        json.dump(training_summary, f, indent=2)
    
    print("\n" + "="*70)
    print("30天数据训练完成！")
    print("="*70)
    print(f"最终测试准确率: {test_metrics['accuracy']:.2%}")
    print(f"F1分数: {test_metrics['f1_score']:.4f}")
    print(f"AUC分数: {test_metrics['auc_score']:.4f}")
    print(f"概率分布: {test_metrics['prob_mean']:.3f} ± {test_metrics['prob_std']:.3f}")
    print("="*70)
    
    return model, test_metrics, history


if __name__ == "__main__":
    main()