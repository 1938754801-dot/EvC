import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.feature_selection import mutual_info_regression
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler, ModelCheckpoint, \
    TensorBoard
import warnings
import os

warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
np.random.seed(42)
tf.random.set_seed(42)


# ============================
# 1. 解决中文显示问题
# ============================
def fix_chinese_font():
    """修复matplotlib中文显示问题"""
    import matplotlib

    # 设置支持中文的字体
    font_list = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'sans-serif']
    matplotlib.rcParams['font.sans-serif'] = font_list
    matplotlib.rcParams['axes.unicode_minus'] = False

    # Windows系统尝试加载字体文件
    if os.name == 'nt':
        font_paths = [
            'C:\\Windows\\Fonts\\simhei.ttf',
            'C:\\Windows\\Fonts\\msyh.ttc',
            'C:\\Windows\\Fonts\\msyh.ttf',
        ]

        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    matplotlib.font_manager.fontManager.addfont(font_path)
                    font_name = matplotlib.font_manager.FontProperties(fname=font_path).get_name()
                    matplotlib.rcParams['font.sans-serif'] = [font_name] + font_list
                    print(f"成功加载字体: {font_name}")
                    break
                except Exception as e:
                    print(f"加载字体失败 {font_path}: {e}")
                    continue


# ============================
# 2. 电池RUL预测器类（改进版-学习容量衰减）
# ============================
class CapacityDecayPredictor:
    def __init__(self, sequence_length=20):  # 增加序列长度到20
        """
        初始化容量衰减预测器

        Parameters:
        -----------
        sequence_length : int, 时间序列长度（增加以学习更长趋势）
        """
        self.sequence_length = sequence_length
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.scaler_y_diff = StandardScaler()  # 新增：用于容量差分的归一化
        self.model = None
        self.feature_importance = {}
        self.predict_diff = True  # 新标志：是否预测容量差分
        print(f"初始化容量衰减预测器，序列长度: {sequence_length}")
        print(f"预测目标: 容量差分（将学习衰减趋势）")

    def load_data(self, file_path):
        """加载原始电池数据"""
        print(f"尝试加载文件: {file_path}")
        try:
            data = pd.read_csv(file_path)
            print(f"数据加载成功！总共有 {len(data)} 行数据")
            print(f"数据形状: {data.shape}")
            print(f"数据列名: {data.columns.tolist()}")

            # 显示前几行数据
            print("\n前5行数据预览:")
            print(data.head())

            return data
        except FileNotFoundError:
            print(f"错误：找不到文件 {file_path}")
            raise
        except Exception as e:
            print(f"加载文件时出错: {e}")
            raise

    def extract_advanced_features_by_cycle(self, data):
        """
        提取高级特征（改进版：使用Q值计算容量并处理异常值）
        """
        print("开始按循环提取高级特征（改进版）...")

        # 确保数据按时间排序
        if 'Time [s]' in data.columns:
            data = data.sort_values('Time [s]')

        # 获取唯一的循环编号
        cycles = sorted(data['Cycle number'].unique())
        print(f"发现 {len(cycles)} 个循环周期")

        features_list = []

        # 第一步：正确提取每个循环的放电容量
        print("第一步：计算每个循环的放电容量...")

        # 存储每个循环的容量值
        cycle_capacities = {}
        cycle_capacities_raw = {}  # 原始计算的容量
        cycle_capacities_robust = {}  # 稳健处理的容量

        for cycle in cycles:
            cycle_data = data[data['Cycle number'] == cycle]

            if len(cycle_data) < 10:  # 跳过数据太少的循环
                continue

            # 方法1：使用Q值计算放电容量（最准确的方法）
            if 'Q [Ah]' in cycle_data.columns and 'Current [mA]' in cycle_data.columns:
                Q_values = cycle_data['Q [Ah]'].values
                current_values = cycle_data['Current [mA]'].values

                # 找到放电阶段（电流为负）
                discharge_mask = current_values < -10  # 使用阈值避免零电流
                if np.sum(discharge_mask) > 5:  # 至少5个放电点
                    discharge_Q = Q_values[discharge_mask]

                    # 放电容量 = 放电开始时的Q值 - 放电结束时的Q值
                    if len(discharge_Q) > 0:
                        discharge_capacity = discharge_Q[0] - discharge_Q[-1]
                        cycle_capacities_raw[cycle] = discharge_capacity
                    else:
                        cycle_capacities_raw[cycle] = 0
                else:
                    # 如果没有明显放电阶段，使用Q的最大值减最小值
                    if len(Q_values) > 0:
                        q_capacity = Q_values.max() - Q_values.min()
                        cycle_capacities_raw[cycle] = q_capacity
                    else:
                        cycle_capacities_raw[cycle] = 0
            else:
                # 方法2：使用Capacity列（备用方法）
                if 'Capacity [Ah]' in cycle_data.columns:
                    cap_values = cycle_data['Capacity [Ah]'].values
                    if len(cap_values) > 0:
                        # 使用最大最小值差作为容量
                        capacity = cap_values.max() - cap_values.min()
                        cycle_capacities_raw[cycle] = capacity
                    else:
                        cycle_capacities_raw[cycle] = 0
                else:
                    cycle_capacities_raw[cycle] = 0

        # 第二步：处理异常容量值
        print("\n第二步：检测和处理异常容量值...")

        # 将原始容量转换为列表（按循环顺序）
        sorted_cycles = sorted(cycle_capacities_raw.keys())
        raw_capacities = [cycle_capacities_raw[cycle] for cycle in sorted_cycles]

        print(f"原始容量统计:")
        print(f"  最小值: {min(raw_capacities):.4f} Ah")
        print(f"  最大值: {max(raw_capacities):.4f} Ah")
        print(f"  平均值: {np.mean(raw_capacities):.4f} Ah")
        print(f"  中位数: {np.median(raw_capacities):.4f} Ah")
        print(f"  标准差: {np.std(raw_capacities):.4f} Ah")

        # 检测异常值（使用IQR方法）
        Q1 = np.percentile(raw_capacities, 25)
        Q3 = np.percentile(raw_capacities, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        print(f"\n异常值检测:")
        print(f"  Q1: {Q1:.4f}, Q3: {Q3:.4f}, IQR: {IQR:.4f}")
        print(f"  正常范围: [{lower_bound:.4f}, {upper_bound:.4f}]")

        # 标记异常值
        outlier_indices = []
        for i, capacity in enumerate(raw_capacities):
            if capacity < lower_bound or capacity > upper_bound:
                outlier_indices.append(i)
                print(f"  循环 {sorted_cycles[i]}: 容量 {capacity:.4f} Ah (异常)")

        print(f"  发现 {len(outlier_indices)} 个异常容量值")

        # 第三步：使用稳健方法处理异常值
        print("\n第三步：处理异常值并平滑容量曲线...")

        # 方法1：使用移动中位数滤波（对异常值更鲁棒）
        window_size = 5
        smoothed_capacities = []

        for i, capacity in enumerate(raw_capacities):
            # 计算窗口内的值
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(raw_capacities), i + window_size // 2 + 1)
            window_values = raw_capacities[start_idx:end_idx]

            # 如果当前点是异常值，使用窗口的中位数
            if i in outlier_indices:
                smoothed_capacity = np.median(window_values)
                print(f"  循环 {sorted_cycles[i]}: 异常值 {capacity:.4f} -> 使用中位数 {smoothed_capacity:.4f}")
            else:
                # 如果不是异常值，使用原始值
                smoothed_capacity = capacity

            smoothed_capacities.append(smoothed_capacity)

        # 方法2：使用移动平均进一步平滑
        smoothed_capacities = pd.Series(smoothed_capacities).rolling(
            window=3, center=True, min_periods=1
        ).mean().tolist()

        # 第四步：提取每个循环的特征
        print("\n第四步：提取每个循环的特征...")

        initial_capacity = None
        previous_capacity = None
        initial_expansion = None
        previous_expansion = None
        capacity_history = []
        capacity_window = 5

        for i, (cycle, smoothed_capacity) in enumerate(zip(sorted_cycles, smoothed_capacities)):
            cycle_data = data[data['Cycle number'] == cycle]

            if len(cycle_data) < 10:  # 跳过数据太少的循环
                continue

            # 获取原始容量用于对比
            raw_capacity = cycle_capacities_raw.get(cycle, 0)

            # 如果平滑后的容量仍异常低，进行额外处理
            if smoothed_capacity < 0.5 * np.median(smoothed_capacities):
                print(f"  警告：循环 {cycle} 的平滑容量 {smoothed_capacity:.4f} 仍过低，使用插值")
                # 使用前后循环的平均值
                if i > 0 and i < len(smoothed_capacities) - 1:
                    smoothed_capacity = (smoothed_capacities[i - 1] + smoothed_capacities[i + 1]) / 2

            feature_dict = {
                'cycle': cycle,
                'capacity': smoothed_capacity,
                'capacity_raw': raw_capacity,  # 保留原始值用于调试
                'is_outlier': 1 if i in outlier_indices else 0  # 标记异常值
            }

            capacity_history.append(smoothed_capacity)

            # 设置初始容量
            if initial_capacity is None:
                initial_capacity = smoothed_capacity
                previous_capacity = smoothed_capacity

            # 容量相关特征
            if initial_capacity and initial_capacity > 0:
                feature_dict.update({
                    'capacity_ratio': smoothed_capacity / initial_capacity,
                    'capacity_loss': initial_capacity - smoothed_capacity,
                    'capacity_loss_ratio': (initial_capacity - smoothed_capacity) / initial_capacity,
                    'remaining_life_ratio': smoothed_capacity / initial_capacity,
                    'decay_state': (initial_capacity - smoothed_capacity) / initial_capacity,
                })

            # 容量变化特征
            if i > 0 and previous_capacity is not None and previous_capacity > 0:
                capacity_change = smoothed_capacity - previous_capacity
                feature_dict['capacity_change'] = capacity_change
                feature_dict['capacity_change_rate'] = capacity_change / previous_capacity

                # 容量加速度
                if i > 1 and len(features_list) > 0:
                    prev_change = features_list[-1].get('capacity_change', 0)
                    feature_dict['capacity_acceleration'] = capacity_change - prev_change
                else:
                    feature_dict['capacity_acceleration'] = 0
            else:
                feature_dict['capacity_change'] = 0
                feature_dict['capacity_change_rate'] = 0
                feature_dict['capacity_acceleration'] = 0

            previous_capacity = smoothed_capacity

            # 容量趋势特征
            if len(capacity_history) >= capacity_window:
                recent_capacities = capacity_history[-capacity_window:]
                x = np.arange(len(recent_capacities))

                try:
                    # 线性趋势
                    coeffs = np.polyfit(x, recent_capacities, 1)
                    feature_dict['capacity_trend_slope'] = coeffs[0]
                    feature_dict['capacity_trend_intercept'] = coeffs[1]

                    # 指数衰减趋势
                    log_capacities = np.log(recent_capacities)
                    exp_coeffs = np.polyfit(x, log_capacities, 1)
                    feature_dict['capacity_exp_decay_rate'] = exp_coeffs[0]
                except:
                    feature_dict['capacity_trend_slope'] = 0
                    feature_dict['capacity_trend_intercept'] = 0
                    feature_dict['capacity_exp_decay_rate'] = 0
            else:
                feature_dict['capacity_trend_slope'] = 0
                feature_dict['capacity_trend_intercept'] = 0
                feature_dict['capacity_exp_decay_rate'] = 0

            # 膨胀相关特征
            if 'Expansion [mu m]' in cycle_data.columns:
                expansion = cycle_data['Expansion [mu m]'].values

                if len(expansion) > 0:
                    if initial_expansion is None:
                        initial_expansion = expansion[0]
                        previous_expansion = expansion[-1]

                    feature_dict.update({
                        'exp_mean': expansion.mean(),
                        'exp_std': expansion.std() if len(expansion) > 1 else 0,
                        'exp_max': expansion.max(),
                        'exp_min': expansion.min(),
                        'exp_range': expansion.max() - expansion.min(),
                        'exp_irreversible': expansion[-1] - initial_expansion,
                        'exp_reversible': expansion.max() - expansion.min(),
                    })

                    if i > 0 and previous_expansion is not None and abs(previous_expansion) > 1e-6:
                        expansion_change = expansion[-1] - previous_expansion
                        feature_dict['exp_change'] = expansion_change
                        feature_dict['exp_change_rate'] = expansion_change / abs(previous_expansion)
                    else:
                        feature_dict['exp_change'] = 0
                        feature_dict['exp_change_rate'] = 0

                    previous_expansion = expansion[-1]

            # 电流相关特征
            if 'Current [mA]' in cycle_data.columns:
                current = cycle_data['Current [mA]'].values
                charge_current = current[current > 0]
                discharge_current = current[current < 0]

                feature_dict.update({
                    'current_mean': current.mean(),
                    'current_abs_mean': np.abs(current).mean(),
                    'charge_current_mean': charge_current.mean() if len(charge_current) > 0 else 0,
                    'discharge_current_mean': np.abs(discharge_current).mean() if len(discharge_current) > 0 else 0,
                    'charge_time_ratio': len(charge_current) / len(current) if len(current) > 0 else 0,
                })

            # 电压相关特征
            if 'Voltage [V]' in cycle_data.columns:
                voltage = cycle_data['Voltage [V]'].values
                feature_dict.update({
                    'voltage_mean': voltage.mean(),
                    'voltage_max': voltage.max(),
                    'voltage_min': voltage.min(),
                    'voltage_range': voltage.max() - voltage.min(),
                    'voltage_std': voltage.std() if len(voltage) > 1 else 0,
                })

            # 温度相关特征
            if 'Temperature [C]' in cycle_data.columns:
                temperature = cycle_data['Temperature [C]'].values
                feature_dict.update({
                    'temp_mean': temperature.mean(),
                    'temp_max': temperature.max(),
                    'temp_min': temperature.min(),
                    'temp_range': temperature.max() - temperature.min(),
                })

            # Q值相关特征
            if 'Q [Ah]' in cycle_data.columns:
                Q = cycle_data['Q [Ah]'].values
                feature_dict.update({
                    'Q_start': Q[0] if len(Q) > 0 else 0,
                    'Q_end': Q[-1] if len(Q) > 0 else 0,
                    'Q_diff': Q[-1] - Q[0] if len(Q) > 0 else 0,
                    'Q_abs_diff': np.abs(Q[-1] - Q[0]) if len(Q) > 0 else 0,
                })

            # 时间相关特征
            if 'Time [s]' in cycle_data.columns:
                time = cycle_data['Time [s]'].values
                if len(time) > 0:
                    duration = time[-1] - time[0]
                    feature_dict['cycle_duration'] = duration
                else:
                    feature_dict['cycle_duration'] = 0

            # 循环计数特征
            feature_dict['cycle_count'] = i
            feature_dict['cycle_ratio'] = i / len(sorted_cycles)

            features_list.append(feature_dict)

            # 进度显示
            if (i + 1) % 20 == 0 or i == 0 or i == len(sorted_cycles) - 1:
                raw_vs_smoothed = f"(原始: {raw_capacity:.4f}, 平滑: {smoothed_capacity:.4f})" if abs(
                    raw_capacity - smoothed_capacity) > 0.01 else ""
                print(f"  已处理 {i + 1}/{len(sorted_cycles)} 个循环 {raw_vs_smoothed}")

        # 创建DataFrame
        features_df = pd.DataFrame(features_list)

        # 确保没有NaN值
        features_df = features_df.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # 验证容量序列
        print(f"\n容量序列验证:")
        print(f"  总循环数: {len(features_df)}")
        print(f"  异常值标记数: {features_df['is_outlier'].sum()}")

        if 'capacity' in features_df.columns:
            capacities = features_df['capacity'].values
            print(f"  平滑后容量统计:")
            print(f"    最小值: {np.min(capacities):.4f} Ah")
            print(f"    最大值: {np.max(capacities):.4f} Ah")
            print(f"    平均值: {np.mean(capacities):.4f} Ah")
            print(f"    中位数: {np.median(capacities):.4f} Ah")
            print(f"    标准差: {np.std(capacities):.4f} Ah")
            print(f"    初始容量: {capacities[0]:.4f} Ah")
            print(f"    最终容量: {capacities[-1]:.4f} Ah")
            print(f"    总衰减: {capacities[0] - capacities[-1]:.4f} Ah")
            print(f"    衰减百分比: {(capacities[0] - capacities[-1]) / capacities[0] * 100:.2f}%")

            # 检查容量单调性
            diff = np.diff(capacities)
            negative_changes = np.sum(diff < -0.1)  # 大幅下降
            positive_changes = np.sum(diff > 0.1)  # 大幅上升
            print(f"    大幅下降次数: {negative_changes}")
            print(f"    大幅上升次数: {positive_changes}")

            # 如果仍有异常，进行最终平滑
            if negative_changes > len(capacities) * 0.1:  # 超过10%的循环有大幅下降
                print("  警告：容量序列仍有较多异常，进行最终平滑...")
                smoothed = pd.Series(capacities).rolling(window=5, center=True, min_periods=1).median()
                features_df['capacity'] = smoothed.values

        print(f"\n高级特征提取完成！总共 {len(features_df)} 个循环，{len(features_df.columns)} 个特征")

        return features_df

    def enhance_capacity_decay_features(self, features_df):
        """
        增强容量衰减相关特征（修正版：正确计算容量差分）
        """
        print("\n增强容量衰减特征...")

        processed_df = features_df.copy()

        # 确保按循环顺序排列
        processed_df = processed_df.sort_values('cycle').reset_index(drop=True)

        # 1. 计算容量差分（一阶差分）
        if 'capacity' in processed_df.columns:
            # 计算相邻循环的容量变化
            processed_df['capacity_diff'] = processed_df['capacity'].diff()

            # 第一个循环没有前一个循环，设置差分为0
            if len(processed_df) > 0:
                processed_df.loc[processed_df.index[0], 'capacity_diff'] = 0

            # 2. 计算容量差分的变化率（二阶差分）
            processed_df['capacity_diff_rate'] = processed_df['capacity_diff'].diff()
            if len(processed_df) > 0:
                processed_df.loc[processed_df.index[0], 'capacity_diff_rate'] = 0
                if len(processed_df) > 1:
                    processed_df.loc[processed_df.index[1], 'capacity_diff_rate'] = 0

            # 3. 计算容量加速度（二阶差分）
            processed_df['capacity_acceleration'] = processed_df['capacity_diff_rate']

            # 4. 计算滚动窗口内的衰减特征
            window_sizes = [3, 5, 10]
            for window in window_sizes:
                # 滚动平均容量
                processed_df[f'capacity_ma_{window}'] = processed_df['capacity'].rolling(window=window,
                                                                                         min_periods=1).mean()
                # 滚动容量差分
                processed_df[f'capacity_diff_ma_{window}'] = processed_df['capacity_diff'].rolling(window=window,
                                                                                                   min_periods=1).mean()
                # 滚动容量衰减率
                processed_df[f'capacity_decay_rate_{window}'] = processed_df['capacity'].pct_change(periods=window)

            # 5. 计算累积衰减特征
            initial_capacity = processed_df['capacity'].iloc[0]
            processed_df['cumulative_decay'] = initial_capacity - processed_df['capacity']
            processed_df['cumulative_decay_rate'] = processed_df[
                                                        'cumulative_decay'] / initial_capacity if initial_capacity > 0 else 0

            # 6. 计算指数衰减拟合残差（检测衰减模式变化）
            if len(processed_df) > 10:
                exp_decay_residuals = []
                for i in range(len(processed_df)):
                    if i >= 5:  # 使用最近5个点
                        start_idx = max(0, i - 4)  # 取5个点：i-4, i-3, i-2, i-1, i
                        window_data = processed_df['capacity'].iloc[start_idx:i + 1].values
                        x = np.arange(len(window_data))
                        try:
                            # 指数拟合 y = a * exp(b*x)
                            log_y = np.log(window_data)
                            coeffs = np.polyfit(x, log_y, 1)
                            predicted = np.exp(coeffs[1] + coeffs[0] * x)
                            residual = np.mean(np.abs(window_data - predicted) / window_data)
                            exp_decay_residuals.append(residual)
                        except:
                            exp_decay_residuals.append(0)
                    else:
                        exp_decay_residuals.append(0)
                processed_df['exp_decay_residual'] = exp_decay_residuals

        # 7. 填充缺失值
        # 先向前填充，再向后填充
        processed_df = processed_df.fillna(method='ffill')
        processed_df = processed_df.fillna(method='bfill')
        processed_df = processed_df.fillna(0)

        # 输出统计信息
        print(
            f"衰减特征增强完成，新增 {len([col for col in processed_df.columns if col not in features_df.columns])} 个衰减特征")

        if 'capacity_diff' in processed_df.columns:
            diff_stats = processed_df['capacity_diff'].describe()
            print(f"容量差分统计:")
            print(f"  平均值: {diff_stats['mean']:.6f} Ah")
            print(f"  标准差: {diff_stats['std']:.6f} Ah")
            print(f"  最小值: {diff_stats['min']:.6f} Ah")
            print(f"  最大值: {diff_stats['max']:.6f} Ah")
            print(
                f"  负值数量: {(processed_df['capacity_diff'] < 0).sum()} ({(processed_df['capacity_diff'] < 0).sum() / len(processed_df) * 100:.1f}%)")
            print(
                f"  正值数量: {(processed_df['capacity_diff'] > 0).sum()} ({(processed_df['capacity_diff'] > 0).sum() / len(processed_df) * 100:.1f}%)")

        return processed_df

    def advanced_feature_selection(self, features_df, target_col='capacity_diff', n_features=15):
        """
        高级特征选择方法（针对容量衰减优化）
        """
        print(f"\n=== 高级特征选择（目标: {target_col}） ===")

        # 1. 准备数据
        # 排除会导致数据泄露的特征
        data_leakage_features = [
            'capacity', 'capacity_ratio', 'capacity_loss',
            'capacity_loss_ratio', 'cumulative_decay', 'decay_percentage',
            'remaining_life_ratio'
        ]

        # 创建排除列表
        exclude_cols = data_leakage_features + ['cycle', 'is_outlier']
        if target_col not in ['capacity', 'capacity_diff']:
            exclude_cols.append(target_col)

        X = features_df.drop(columns=[col for col in exclude_cols if col in features_df.columns], errors='ignore')

        # 设置目标变量
        if target_col == 'capacity_diff':
            y = features_df['capacity'].diff().fillna(0)
        else:
            y = features_df[target_col]

        # 只选择数值型特征
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_numeric = X[numeric_cols]

        # 填充缺失值
        X_numeric = X_numeric.fillna(0)
        y = y.fillna(0)

        # 2. 互信息法选择特征
        print("计算特征互信息（针对容量衰减）...")
        try:
            mi_scores = mutual_info_regression(X_numeric, y, random_state=42)
            mi_scores = pd.Series(mi_scores, index=X_numeric.columns)
            mi_scores = mi_scores.sort_values(ascending=False)

            print("特征重要性排序（互信息）- 前20个:")
            for i, (feature, score) in enumerate(mi_scores.head(20).items()):
                print(f"{i + 1:2d}. {feature:30s}: {score:.6f}")
        except Exception as e:
            print(f"互信息计算失败: {e}")
            mi_scores = pd.Series(0, index=X_numeric.columns)

        # 3. 基于相关性的特征选择
        correlation_scores = []
        for col in X_numeric.columns:
            try:
                corr = np.abs(np.corrcoef(X_numeric[col].fillna(0), y)[0, 1])
                if not np.isnan(corr):
                    correlation_scores.append((col, corr))
            except:
                continue

        correlation_scores.sort(key=lambda x: x[1], reverse=True)

        print("\n特征相关性排序 - 前15个:")
        for i, (feature, corr) in enumerate(correlation_scores[:15]):
            print(f"{i + 1:2d}. {feature:30s}: {corr:.4f}")

        # 4. 结合两种方法选择特征
        # 选择互信息前n_features*2的特征
        top_mi_features = mi_scores.head(n_features * 2).index.tolist()
        # 选择相关性前n_features*2的特征
        top_corr_features = [f for f, _ in correlation_scores[:n_features * 2]]

        # 取交集
        common_features = list(set(top_mi_features) & set(top_corr_features))

        print(f"\n共同特征数量: {len(common_features)}")

        # 如果共同特征足够，使用共同特征
        if len(common_features) >= min(5, n_features):
            selected_features = common_features[:n_features]
            print("使用共同特征作为最终选择")
        else:
            # 如果共同特征不足，使用加权得分
            print("共同特征不足，使用加权得分选择...")
            feature_scores = {}
            all_candidate_features = set(top_mi_features + top_corr_features)

            for feature in all_candidate_features:
                mi_score = mi_scores.get(feature, 0)
                corr_score = dict(correlation_scores).get(feature, 0)
                # 加权得分：互信息权重0.6，相关性权重0.4
                feature_scores[feature] = mi_score * 0.6 + corr_score * 0.4

            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f for f, _ in sorted_features[:n_features]]

        # 5. 确保包含一些关键的衰减特征
        key_decay_features = ['capacity_trend_slope', 'capacity_exp_decay_rate',
                              'capacity_diff_ma_5', 'capacity_decay_rate_5',
                              'exp_irreversible', 'exp_reversible',
                              'cycle_count', 'cycle_ratio', 'decay_state']

        # 添加尚未包含的关键衰减特征
        added_count = 0
        for feature in key_decay_features:
            if (feature in X_numeric.columns and
                    feature not in selected_features and
                    added_count < 4 and  # 最多添加4个
                    len(selected_features) < n_features):
                selected_features.append(feature)
                added_count += 1

        # 6. 移除可能导致问题的特征
        final_selected_features = []
        for feature in selected_features:
            # 避免选择标准差非常小的特征（可能没有信息量）
            if feature in X_numeric.columns:
                if X_numeric[feature].std() > 1e-6:  # 排除几乎不变的特征
                    final_selected_features.append(feature)

        # 如果特征太少，添加一些额外的特征
        if len(final_selected_features) < 8:
            print("选择的特征太少，添加额外特征...")
            for feature in X_numeric.columns:
                if (feature not in final_selected_features and
                        feature not in data_leakage_features and
                        X_numeric[feature].std() > 1e-6):
                    final_selected_features.append(feature)
                    if len(final_selected_features) >= 10:
                        break

        print(f"\n最终选择的 {len(final_selected_features)} 个特征:")
        for i, feature in enumerate(final_selected_features):
            # 显示特征的相关性和互信息分数
            mi_score = mi_scores.get(feature, 0)
            corr_score = dict(correlation_scores).get(feature, 0)
            print(f"  {i + 1:2d}. {feature:30s} [MI: {mi_score:.4f}, Corr: {corr_score:.4f}]")

        return final_selected_features

    def prepare_sequences_for_decay_prediction(self, features_df, feature_cols, predict_diff=True):
        """
        准备时间序列数据（修正版：正确处理容量差分）
        """
        print(f"\n准备时间序列数据（序列长度: {self.sequence_length}）")
        print(f"预测目标: {'容量差分' if predict_diff else '容量绝对值'}")

        # 确保特征列存在
        available_cols = []
        for col in feature_cols:
            if col in features_df.columns:
                available_cols.append(col)
            else:
                print(f"警告：特征列 '{col}' 不存在，已跳过")

        if len(available_cols) == 0:
            print("错误：没有可用的特征列！")
            # 使用一些基本的数值型特征
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
            # 排除容量相关的列，避免数据泄露
            exclude_cols = ['capacity', 'cycle', 'capacity_diff']
            available_cols = [col for col in numeric_cols if col not in exclude_cols]
            print(f"使用默认特征: {available_cols[:10]}...")

        print(f"使用 {len(available_cols)} 个特征")

        # 提取特征
        X = features_df[available_cols].values

        # 根据预测目标准备y值
        if predict_diff:
            # 预测容量差分
            if 'capacity_diff' in features_df.columns:
                y = features_df['capacity_diff'].values.reshape(-1, 1)
            else:
                # 如果不存在，计算容量差分
                y = features_df['capacity'].diff().values.reshape(-1, 1)
                y[0] = 0  # 设置第一个值为0

            # 检查y的值
            print(f"容量差分统计:")
            print(f"  形状: {y.shape}")
            print(f"  非零值数量: {np.count_nonzero(y)}")
            print(f"  平均值: {np.mean(y):.6f}")
            print(f"  标准差: {np.std(y):.6f}")

            # 使用差分归一化器
            y_scaled = self.scaler_y_diff.fit_transform(y)
            print(f"预测目标: 容量差分（一阶差分）")
        else:
            # 预测容量绝对值
            y = features_df['capacity'].values.reshape(-1, 1)
            y_scaled = self.scaler_y.fit_transform(y)
            print(f"预测目标: 容量绝对值")

        # 归一化特征
        X_scaled = self.scaler_X.fit_transform(X)

        print(f"原始特征形状: {X.shape}, 目标形状: {y.shape}")

        # 创建时间序列
        X_seq, y_seq = [], []

        for i in range(len(X_scaled) - self.sequence_length):
            X_seq.append(X_scaled[i:i + self.sequence_length])
            y_seq.append(y_scaled[i + self.sequence_length])

        if len(X_seq) == 0:
            print(f"错误：序列长度为 {self.sequence_length} 时无法创建序列")
            print(f"数据长度: {len(X_scaled)}，尝试减小序列长度...")

            # 尝试减小序列长度
            original_length = self.sequence_length
            self.sequence_length = min(10, len(X_scaled) // 2)
            print(f"将序列长度从 {original_length} 减小到 {self.sequence_length}")

            X_seq, y_seq = [], []
            for i in range(len(X_scaled) - self.sequence_length):
                X_seq.append(X_scaled[i:i + self.sequence_length])
                y_seq.append(y_scaled[i + self.sequence_length])

        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)

        print(f"创建了 {len(X_seq)} 个序列，每个序列长度为 {self.sequence_length}")
        print(f"输入形状: {X_seq.shape}, 输出形状: {y_seq.shape}")

        return X_seq, y_seq

    def build_decay_prediction_model(self, input_shape, dropout_rate=0.3):
        """
        构建容量衰减预测模型（简化版，删除注意力机制）
        """
        print(f"构建容量衰减预测模型，输入形状: {input_shape}")

        inputs = keras.Input(shape=input_shape)

        # 第一层LSTM - 捕捉长期依赖
        x = layers.LSTM(128, return_sequences=True,
                        kernel_regularizer=keras.regularizers.l2(0.01),
                        dropout=0.2,
                        recurrent_dropout=0.2)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

        # 第二层LSTM - 提取特征
        x = layers.LSTM(64, return_sequences=False,  # 改为return_sequences=False
                        kernel_regularizer=keras.regularizers.l2(0.01),
                        dropout=0.2,
                        recurrent_dropout=0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate)(x)

        # 第三层LSTM - 输出编码（如果需要更多层，可以保留）
        # x = layers.LSTM(32, return_sequences=False,
        #                 kernel_regularizer=keras.regularizers.l2(0.01),
        #                 dropout=0.1,
        #                 recurrent_dropout=0.1)(x)
        # x = layers.BatchNormalization()(x)

        # 删除注意力机制，简化模型结构
        # attention = layers.Dense(input_shape[0], activation='softmax')(x)
        # x = layers.Multiply()([x, attention])

        # 密集层
        x = layers.Dense(32, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout_rate * 0.5)(x)

        x = layers.Dense(16, activation='relu',
                         kernel_regularizer=keras.regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)

        # 输出层 - 预测容量差分
        outputs = layers.Dense(1)(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        # 使用Adam优化器
        optimizer = keras.optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )

        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )

        print("容量衰减预测模型构建完成（简化版）")
        return model

    def train_decay_model(self, X_train, y_train, X_val, y_val, epochs=150):
        """
        训练容量衰减模型
        """
        print("训练容量衰减模型...")

        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=40,
                restore_best_weights=True,
                min_delta=1e-4,
                verbose=1,
                mode='min'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=20,
                min_lr=1e-6,
                verbose=1,
                mode='min'
            ),
            ModelCheckpoint(
                'best_decay_model.keras',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]

        # 调整批次大小
        batch_size = min(32, len(X_train) // 10)

        print(f"训练参数: epochs={epochs}, batch_size={batch_size}")

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        # 分析训练过程
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]

        print("\n=== 训练分析 ===")
        print(f"最终训练损失: {final_train_loss:.6f}")
        print(f"最终验证损失: {final_val_loss:.6f}")

        if final_train_loss > 0:
            overfit_ratio = final_val_loss / final_train_loss
            print(f"过拟合程度: {overfit_ratio:.2f}x")

        return history

    def predict_with_decay_model(self, X, predict_diff=True):
        """
        使用衰减模型进行预测
        """
        y_pred_scaled = self.model.predict(X, verbose=0)

        if predict_diff:
            # 如果是预测差分，需要逆变换差分值
            y_pred_diff = self.scaler_y_diff.inverse_transform(y_pred_scaled)
            return y_pred_diff
        else:
            # 如果是预测绝对值，直接逆变换
            y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
            return y_pred

    def predict_capacity_from_diff(self, X_seq, initial_capacities):
        """
        从容量差分预测还原容量绝对值

        Parameters:
        -----------
        X_seq : 输入序列
        initial_capacities : 每个序列的初始容量值
        """
        # 预测容量差分
        y_pred_diff = self.predict_with_decay_model(X_seq, predict_diff=True)

        # 将差分转换为容量绝对值
        y_pred_capacity = []
        for i in range(len(y_pred_diff)):
            if i < len(initial_capacities):
                # 初始容量 + 累积差分
                pred_capacity = initial_capacities[i] + np.cumsum(y_pred_diff[i:i + 1])[-1]
                y_pred_capacity.append(pred_capacity)
            else:
                # 如果没有初始容量，使用最后一个已知容量
                y_pred_capacity.append(y_pred_capacity[-1] if y_pred_capacity else 0)

        return np.array(y_pred_capacity).reshape(-1, 1)

    def evaluate_decay_prediction(self, X_test, y_test_original, features_df, test_indices):
        """
        评估衰减预测性能
        """
        print("\n=== 容量衰减预测评估 ===")

        # 预测容量差分
        y_pred_diff = self.predict_with_decay_model(X_test, predict_diff=True)

        # 计算真实的容量差分
        y_true_diff = np.diff(features_df['capacity'].values[test_indices])
        # 对齐长度
        y_true_diff = np.concatenate([[0], y_true_diff])[:len(y_pred_diff)]

        # 评估差分预测
        diff_metrics = self.calculate_metrics(y_true_diff.reshape(-1, 1), y_pred_diff)
        print("\n容量差分预测指标:")
        for metric, value in diff_metrics.items():
            if metric == 'MAPE':
                print(f"  {metric}: {value:.2f}%")
            elif metric == 'R2':
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value:.4f}")

        # 从差分还原容量预测
        # 获取测试集对应的初始容量
        test_start_idx = test_indices[0] if len(test_indices) > 0 else 0
        initial_capacities_for_test = []

        for i in range(len(X_test)):
            # 每个测试序列的初始容量是序列开始前一个时间步的容量
            seq_start_idx = test_start_idx + i
            if seq_start_idx > 0:
                initial_capacity = features_df['capacity'].iloc[seq_start_idx - 1]
            else:
                initial_capacity = features_df['capacity'].iloc[0]
            initial_capacities_for_test.append(initial_capacity)

        # 预测容量绝对值
        y_pred_capacity = self.predict_capacity_from_diff(X_test, initial_capacities_for_test)

        # 评估容量绝对值预测
        capacity_metrics = self.calculate_metrics(y_test_original, y_pred_capacity)
        print("\n容量绝对值预测指标:")
        for metric, value in capacity_metrics.items():
            if metric == 'MAPE':
                print(f"  {metric}: {value:.2f}%")
            elif metric == 'R2':
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value:.4f}")

        return y_pred_diff, y_pred_capacity, diff_metrics, capacity_metrics

    def calculate_metrics(self, y_true, y_pred):
        """计算评估指标"""
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        # 避免除以0的情况
        mask = y_true != 0
        if np.any(mask):
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = float('inf')

        # 计算R²分数
        r2 = r2_score(y_true, y_pred)

        return {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R2': r2
        }


# ============================
# 3. 可视化函数（增强版）
# ============================
def plot_capacity_decay_analysis(features_df, title="容量衰减分析"):
    """绘制容量衰减分析图"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. 容量衰减曲线
    axes[0, 0].plot(features_df['cycle'], features_df['capacity'], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('循环次数')
    axes[0, 0].set_ylabel('容量 (Ah)')
    axes[0, 0].set_title('容量衰减曲线')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 容量差分（衰减率）
    if 'capacity_diff' in features_df.columns:
        axes[0, 1].plot(features_df['cycle'][1:], features_df['capacity_diff'][1:], 'r-', linewidth=1.5)
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('循环次数')
        axes[0, 1].set_ylabel('容量变化 (Ah)')
        axes[0, 1].set_title('容量变化（差分）')
        axes[0, 1].grid(True, alpha=0.3)

    # 3. 累积衰减
    if 'cumulative_decay' in features_df.columns:
        axes[0, 2].plot(features_df['cycle'], features_df['cumulative_decay'], 'g-', linewidth=2)
        axes[0, 2].set_xlabel('循环次数')
        axes[0, 2].set_ylabel('累积衰减 (Ah)')
        axes[0, 2].set_title('累积容量衰减')
        axes[0, 2].grid(True, alpha=0.3)

    # 4. 容量衰减率
    if 'capacity_change_rate' in features_df.columns:
        axes[1, 0].plot(features_df['cycle'][1:], features_df['capacity_change_rate'][1:], color='purple',
                        linestyle='-', linewidth=1.5)
        axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 0].set_xlabel('循环次数')
        axes[1, 0].set_ylabel('衰减率')
        axes[1, 0].set_title('容量衰减率')
        axes[1, 0].grid(True, alpha=0.3)

    # 5. 容量与膨胀关系
    if 'exp_irreversible' in features_df.columns:
        axes[1, 1].scatter(features_df['exp_irreversible'], features_df['capacity'], alpha=0.6, s=20)
        axes[1, 1].set_xlabel('不可逆膨胀 (μm)')
        axes[1, 1].set_ylabel('容量 (Ah)')
        axes[1, 1].set_title('容量 vs 不可逆膨胀')
        axes[1, 1].grid(True, alpha=0.3)

    # 6. 衰减趋势斜率
    if 'capacity_trend_slope' in features_df.columns:
        axes[1, 2].plot(features_df['cycle'][5:], features_df['capacity_trend_slope'][5:], color='orange',
                        linestyle='-', linewidth=1.5)
        axes[1, 2].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlabel('循环次数')
        axes[1, 2].set_ylabel('趋势斜率')
        axes[1, 2].set_title('容量衰减趋势斜率（负值表示衰减）')
        axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_decay_prediction_comparison(y_true, y_pred, y_true_diff=None, y_pred_diff=None,
                                     title="容量衰减预测结果"):
    """绘制容量衰减预测对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 容量绝对值预测对比
    axes[0, 0].plot(y_true, 'b-', label='真实容量', linewidth=2, alpha=0.8)
    axes[0, 0].plot(y_pred, 'r--', label='预测容量', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('样本索引')
    axes[0, 0].set_ylabel('容量 (Ah)')
    axes[0, 0].set_title('容量绝对值预测')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. 容量差分预测对比
    if y_true_diff is not None and y_pred_diff is not None:
        axes[0, 1].plot(y_true_diff, 'g-', label='真实容量变化', linewidth=1.5, alpha=0.8)
        axes[0, 1].plot(y_pred_diff, 'm--', label='预测容量变化', linewidth=1.5, alpha=0.8)
        axes[0, 1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        axes[0, 1].set_xlabel('样本索引')
        axes[0, 1].set_ylabel('容量变化 (Ah)')
        axes[0, 1].set_title('容量变化（差分）预测')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. 预测误差
    errors = y_true - y_pred
    axes[1, 0].plot(errors, 'r-', linewidth=1, alpha=0.7)
    axes[1, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('样本索引')
    axes[1, 0].set_ylabel('预测误差 (Ah)')
    axes[1, 0].set_title('容量预测误差')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. 误差分布直方图
    axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_xlabel('预测误差 (Ah)')
    axes[1, 1].set_ylabel('频次')
    axes[1, 1].set_title('预测误差分布')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_training_history(history):
    """绘制训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 绘制损失曲线
    axes[0, 0].plot(history.history['loss'], label='训练损失')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='验证损失')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('训练和验证损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 绘制MAE曲线
    axes[0, 1].plot(history.history['mae'], label='训练MAE')
    if 'val_mae' in history.history:
        axes[0, 1].plot(history.history['val_mae'], label='验证MAE')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('训练和验证MAE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 绘制学习率曲线
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], label='学习率')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_title('学习率变化')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 绘制RMSE曲线
    if 'root_mean_squared_error' in history.history:
        axes[1, 1].plot(history.history['root_mean_squared_error'], label='训练RMSE')
        if 'val_root_mean_squared_error' in history.history:
            axes[1, 1].plot(history.history['val_root_mean_squared_error'], label='验证RMSE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].set_title('训练和验证RMSE')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================
# 4. 主程序（容量衰减预测版本）
# ============================
def main():
    """主程序"""
    print("=== 容量衰减预测系统 ===")
    print("目标：学习容量衰减趋势，而非平均容量")
    print(f"序列长度：20（更长历史窗口）")
    print("预测方式：先预测容量差分，再转换为容量绝对值\n")

    # 修复中文显示
    fix_chinese_font()

    # 1. 初始化容量衰减预测器
    predictor = CapacityDecayPredictor(sequence_length=20)  # 使用更长序列

    # 2. 加载数据
    file_path = r"C:\Users\86176\Desktop\研一上\多传感\选题4参考文献和数据集\1. 锂离子多物理场数据\data\data\21\cycling_wExpansion.csv"
    data = predictor.load_data(file_path)

    # 3. 提取高级特征
    print("\n" + "=" * 60)
    print("特征提取阶段")
    print("=" * 60)
    features_df = predictor.extract_advanced_features_by_cycle(data)

    # 4. 增强容量衰减特征
    features_df = predictor.enhance_capacity_decay_features(features_df)

    # 5. 绘制容量衰减分析图
    plot_capacity_decay_analysis(features_df, title="原始数据容量衰减分析")

    # 6. 高级特征选择（针对容量衰减）
    print("\n" + "=" * 60)
    print("特征选择（针对容量衰减优化）")
    print("=" * 60)
    selected_features = predictor.advanced_feature_selection(features_df, target_col='capacity_diff', n_features=15)

    # 7. 准备序列数据（预测容量差分）
    print("\n" + "=" * 60)
    print("数据准备阶段（预测容量差分）")
    print("=" * 60)

    try:
        # 准备预测容量差分的序列数据
        X_seq, y_seq = predictor.prepare_sequences_for_decay_prediction(
            features_df, selected_features, predict_diff=True
        )
    except Exception as e:
        print(f"准备序列数据失败: {e}")
        # 尝试减小序列长度
        predictor.sequence_length = 15
        X_seq, y_seq = predictor.prepare_sequences_for_decay_prediction(
            features_df, selected_features, predict_diff=True
        )

    # 8. 数据划分
    train_size = int(len(X_seq) * 0.7)
    val_size = int(len(X_seq) * 0.15)

    X_train = X_seq[:train_size]
    X_val = X_seq[train_size:train_size + val_size]
    X_test = X_seq[train_size + val_size:]

    y_train = y_seq[:train_size]
    y_val = y_seq[train_size:train_size + val_size]
    y_test = y_seq[train_size + val_size:]

    # 获取测试集索引（用于后续分析）
    test_indices = list(range(train_size + val_size, len(X_seq)))

    print(f"\n数据划分结果:")
    print(f"  训练集: {len(X_train)} 个样本 ({len(X_train) / len(X_seq) * 100:.1f}%)")
    print(f"  验证集: {len(X_val)} 个样本 ({len(X_val) / len(X_seq) * 100:.1f}%)")
    print(f"  测试集: {len(X_test)} 个样本 ({len(X_test) / len(X_seq) * 100:.1f}%)")

    # 9. 构建容量衰减预测模型
    print("\n" + "=" * 60)
    print("模型构建阶段（容量衰减预测）")
    print("=" * 60)

    predictor.model = predictor.build_decay_prediction_model((X_train.shape[1], X_train.shape[2]))

    # 打印模型摘要
    print("\n模型结构摘要:")
    predictor.model.summary()

    # 10. 训练容量衰减模型
    print("\n" + "=" * 60)
    print("模型训练阶段")
    print("=" * 60)

    history = predictor.train_decay_model(X_train, y_train, X_val, y_val, epochs=150)

    # 11. 绘制训练历史
    plot_training_history(history)

    # 12. 评估模型
    print("\n" + "=" * 60)
    print("模型评估阶段")
    print("=" * 60)

    # 获取测试集真实容量值
    y_test_original = features_df['capacity'].values[test_indices].reshape(-1, 1)

    # 评估衰减预测
    y_pred_diff, y_pred_capacity, diff_metrics, capacity_metrics = predictor.evaluate_decay_prediction(
        X_test, y_test_original, features_df, test_indices
    )

    # 13. 绘制预测结果
    # 获取真实的容量差分（用于对比）
    y_true_diff = np.diff(features_df['capacity'].values[test_indices])
    y_true_diff = np.concatenate([[0], y_true_diff])[:len(y_pred_diff)]

    plot_decay_prediction_comparison(
        y_test_original.flatten(),
        y_pred_capacity.flatten(),
        y_true_diff,
        y_pred_diff.flatten(),
        title="容量衰减预测结果对比"
    )

    # 14. 保存结果
    print("\n" + "=" * 60)
    print("保存结果")
    print("=" * 60)

    try:
        # 保存特征数据
        features_df.to_csv('decay_features.csv', index=False)
        print("特征数据已保存到: decay_features.csv")

        # 保存模型
        predictor.model.save('capacity_decay_model.keras')
        print("模型已保存到: capacity_decay_model.keras")

        # 保存评估结果
        with open('decay_prediction_results.txt', 'w') as f:
            f.write("=== 容量衰减预测评估结果 ===\n\n")
            f.write("容量差分预测指标:\n")
            for metric, value in diff_metrics.items():
                f.write(f"  {metric}: {value}\n")

            f.write("\n容量绝对值预测指标:\n")
            for metric, value in capacity_metrics.items():
                f.write(f"  {metric}: {value}\n")

            f.write(f"\n序列长度: {predictor.sequence_length}\n")
            f.write(f"预测方式: 容量差分 -> 容量绝对值\n")
            f.write(f"特征数量: {len(selected_features)}\n")

        print("评估结果已保存到: decay_prediction_results.txt")

        # 保存选择的特征
        with open('decay_selected_features.txt', 'w') as f:
            f.write("=== 容量衰减预测选择的特征 ===\n\n")
            for i, feature in enumerate(selected_features):
                f.write(f"{i + 1}. {feature}\n")

        print("选择的特征已保存到: decay_selected_features.txt")

        # 保存预测结果
        results_df = pd.DataFrame({
            'true_capacity': y_test_original.flatten(),
            'predicted_capacity': y_pred_capacity.flatten(),
            'true_capacity_diff': y_true_diff,
            'predicted_capacity_diff': y_pred_diff.flatten()
        })
        results_df.to_csv('decay_predictions.csv', index=False)
        print("预测结果已保存到: decay_predictions.csv")

    except Exception as e:
        print(f"保存结果时出错: {e}")

    print("\n程序运行完成！")
    print("模型现在专注于学习容量衰减趋势，使用20个时间步的历史窗口。")
    print("预测流程：先预测容量差分，再转换为容量绝对值。")


# ============================
# 5. 程序入口
# ============================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback

        traceback.print_exc()