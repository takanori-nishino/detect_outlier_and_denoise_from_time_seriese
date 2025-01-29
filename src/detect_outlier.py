import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import seaborn as sns
from datetime import datetime
import os
import json

class CoordinateAnomalyDetector:
    def __init__(self, window_size=1, threshold_sigma=3):
        self.window_size = window_size
        self.threshold_sigma = threshold_sigma
        self.scaler = StandardScaler()
        self.model = self._build_model()
        self.history = None

    def save_train_data(self, train_data, output_dir):
        """学習データをCSVファイルとして保存"""
        train_df = pd.DataFrame(
            train_data,
            columns=['forward_component', 'lateral_component', 'magnitude']
        )
        # エスケープ文字を指定して保存
        train_df.to_csv(os.path.join(output_dir, 'train_data.csv'), 
                        index=False, escapechar='\\')

    def save_test_results(self, test_coords, test_times, test_data, anomalies, output_dir):
        """検証データと結果をCSVファイルとして保存"""
        # window_sizeの影響で長さが異なる場合の調整
        valid_length = min(len(test_coords) - self.window_size, len(anomalies))
        
        results_df = pd.DataFrame({
            'time': test_times[self.window_size:self.window_size + valid_length],
            'x': test_coords[self.window_size:self.window_size + valid_length, 0],
            'y': test_coords[self.window_size:self.window_size + valid_length, 1],
            'vector_magnitude': np.linalg.norm(
                np.diff(test_coords[self.window_size-1:self.window_size + valid_length], axis=0),
                axis=1
            ),
            'is_anomaly': anomalies[:valid_length]
        })
        
        results_df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
        print(f"検証結果を保存しました: {os.path.join(output_dir, 'test_results.csv')}")

    def save_model(self, output_dir):
        """モデルと関連するパラメータを保存"""
        # モデルの保存（.keras形式）
        model_path = os.path.join(output_dir, 'model.keras')
        self.model.save(model_path)
        
        # スケーラーの保存
        import joblib
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(self.scaler, scaler_path)
        
        # パラメータの保存
        params = {
            'window_size': self.window_size,
            'threshold_sigma': self.threshold_sigma
        }
        params_path = os.path.join(output_dir, 'parameters.json')
        with open(params_path, 'w') as f:
            json.dump(params, f)
        
        print(f"モデルを保存しました: {model_path}")
        print(f"スケーラーを保存しました: {scaler_path}")
        print(f"パラメータを保存しました: {params_path}")

    def _convert_to_relative_vectors(self, velocity):
        """
        速度ベクトルを相対表現に変換します。

        Returns:
            data: [進行方向成分の大きさ, 進行方向と直行する成分の大きさ, ベクトルの大きさ] の形式の配列
        """
        magnitudes = np.linalg.norm(velocity, axis=1)  # 各ベクトルの大きさ (n_samples,)
        data = np.zeros((len(velocity), 3))  # (n_samples, 3)

        for i in range(1, len(velocity)):
            prev_vec = velocity[i - 1]
            curr_vec = velocity[i]

            prev_mag = np.linalg.norm(prev_vec)
            curr_mag = magnitudes[i]

            if prev_mag < 1e-6 or curr_mag < 1e-6:
                # ベクトルの大きさが0に近い場合はスキップ
                continue

            # 前のベクトルの方向を基準とした単位ベクトル
            prev_dir = prev_vec / prev_mag

            # 進行方向成分（前のベクトルの方向への投影）
            forward_component = np.dot(curr_vec, prev_dir)

            # 進行方向と直行する成分（前のベクトルの方向に垂直な方向への投影）
            lateral_component = np.linalg.norm(curr_vec - forward_component * prev_dir)

            # データに格納
            data[i] = [forward_component, lateral_component, curr_mag]

        # 最初のデータ点を補完（次のデータを使用）
        data[0] = data[1]
        return data

    def _build_model(self):
        class VectorComponentLayer(tf.keras.layers.Layer):
            def __init__(self, window_size, **kwargs):
                super().__init__(**kwargs)
                self.window_size = window_size

                # シンプルな2層構造
                self.common = tf.keras.layers.Dense(8, activation='linear')
                self.output_layer = tf.keras.layers.Dense(3)  # 出力を3次元に変更

            def call(self, inputs):
                # 入力を整形
                sequence = tf.reshape(inputs, (-1, self.window_size * 3))
                # 線形変換
                features = self.common(sequence)
                # 出力（制約なし）
                outputs = self.output_layer(features)
                return outputs

        inputs = tf.keras.Input(shape=(self.window_size * 3,))
        outputs = VectorComponentLayer(self.window_size)(inputs)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                      loss='mse')

        return model

    def fit(self, train_data, epochs=10, batch_size=32, validation_split=0.2):
        scaled_data = self.scaler.fit_transform(train_data)
        X, y = self._create_sequences(scaled_data)

        self.history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

    def _calculate_velocity(self, coords):
        velocity = np.diff(coords, axis=0)
        # 先頭にゼロベクトルを追加して元の長さに合わせる
        velocity = np.vstack([np.zeros((1, 2)), velocity])
        return velocity

    def _preprocess_velocity(self, velocity):
        data = self._convert_to_relative_vectors(velocity)
        # Noneをゼロで置換（または他の適切な値で補完）
        data = np.nan_to_num(data)
        return data

    @classmethod
    def load_model(cls, model_dir):
        """保存されたモデルとパラメータを読み込む"""
        # パラメータの読み込み
        with open(os.path.join(model_dir, 'parameters.json'), 'r') as f:
            params = json.load(f)
        
        # インスタンスの作成
        detector = cls(
            window_size=params['window_size'],
            threshold_sigma=params['threshold_sigma']
        )
        
        # モデルの読み込み
        model_path = os.path.join(model_dir, 'model.keras')
        detector.model = tf.keras.models.load_model(model_path)
        
        # スケーラーの読み込み
        import joblib
        scaler_path = os.path.join(model_dir, 'scaler.joblib')
        detector.scaler = joblib.load(scaler_path)
        
        return detector
        
    def load_and_process_data(self, csv_path):
        """データの読み込みと前処理"""
        # まず文字列として読み込む
        df = pd.read_csv(csv_path, low_memory=False)
        
        def clean_numeric_column(series):
            # 'None'をNaNに置換
            series = series.replace('None', np.nan)
            # 不正な文字が含まれている場合はNaNに置換
            series = pd.to_numeric(series, errors='coerce')
            return series
        
        # 各カラムをクリーニング
        df['norm_arena_x'] = clean_numeric_column(df['norm_arena_x'])
        df['norm_arena_y'] = clean_numeric_column(df['norm_arena_y'])
        df['time'] = clean_numeric_column(df['time'])
        
        # 必要なカラムの欠損値を確認
        nan_count = df[['time', 'norm_arena_x', 'norm_arena_y']].isna().sum()
        print(f"NaN count before interpolation - time: {nan_count['time']}, "
            f"x: {nan_count['norm_arena_x']}, y: {nan_count['norm_arena_y']}")
        
        # 線形補間を実行
        df[['norm_arena_x', 'norm_arena_y']] = df[['norm_arena_x', 'norm_arena_y']].interpolate(method='linear')
        
        # 補間後も残っているNaNがあれば（端の値など）、前後の値で埋める
        df[['norm_arena_x', 'norm_arena_y']] = (df[['norm_arena_x', 'norm_arena_y']]
                                            .interpolate(method='linear', limit_direction='both'))
        
        # 座標とタイムスタンプを取得
        coords = df[['norm_arena_x', 'norm_arena_y']].values
        times = df['time'].values

        velocity = self._calculate_velocity(coords)
        magnitudes = np.linalg.norm(velocity, axis=1)
        prev_magnitudes = np.zeros_like(magnitudes)
        prev_magnitudes[1:] = magnitudes[:-1]
        processed_data = self._preprocess_velocity(velocity)
        valid_prev_magnitude = prev_magnitudes < 0.3

        train_mask = np.zeros_like(times, dtype=bool)
        for t in [0.2, 0.4, 0.6, 0.8]:
            train_mask = train_mask | np.isclose(times % 1, t, atol=1e-10)
        
        test_mask = np.isclose(times % 1, 0.0, atol=1e-10)
        train_valid_mask = train_mask & valid_prev_magnitude

        train_data = processed_data[train_valid_mask]
        test_data = processed_data[test_mask]
        test_coords = coords[test_mask]
        test_times = times[test_mask]

        return train_data, test_data, test_coords, test_times, df

    def _create_sequences(self, data):
        """シーケンスデータの作成"""
        X, y = [], []
        for i in range(len(data) - self.window_size):
            sequence = data[i:i + self.window_size]
            next_vector = data[i + self.window_size]
            X.append(sequence.flatten())
            y.append(next_vector)
        return np.array(X), np.array(y)

    def detect_anomalies(self, test_data):
        scaled_data = self.scaler.transform(test_data)
        X, y_true = self._create_sequences(scaled_data)
        y_pred = self.model.predict(X)

        # 誤差の計算
        errors = np.linalg.norm(y_true - y_pred, axis=1)
        threshold = np.mean(errors) + self.threshold_sigma * np.std(errors)
        anomalies = errors > threshold

        return anomalies, errors, threshold, y_true, y_pred, y_true - y_pred

    def visualize_results(self, test_coords, test_data, test_times, anomalies, errors, threshold,
                        y_true_original, y_pred_original, errors_by_vector):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"velocity_anomaly_detection_results_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # サイズの不一致を修正
        valid_length = min(len(test_coords) - self.window_size, len(anomalies))
        plot_anomalies = anomalies[:valid_length]
        plot_coords = test_coords[self.window_size:self.window_size + valid_length]
        plot_data = test_data[self.window_size:self.window_size + valid_length]
        plot_times = test_times[self.window_size:self.window_size + valid_length]

        # 1. 学習履歴のプロット
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Learning History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{output_dir}/learning_history.png")
        plt.close()

        # 2. 軌跡プロット（異常値をハイライト）
        plt.figure(figsize=(10, 8))
        plt.scatter(plot_coords[~plot_anomalies, 0],
                    plot_coords[~plot_anomalies, 1],
                    c='blue', label='Normal', alpha=0.5)
        plt.scatter(plot_coords[plot_anomalies, 0],
                    plot_coords[plot_anomalies, 1],
                    c='red', label='Anomaly', alpha=0.7)
        plt.title('Coordinate Trajectory with Anomalies')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.legend()
        plt.savefig(f"{output_dir}/trajectory_with_anomalies.png")
        plt.close()

        # 3. 時系列での誤差プロット
        plt.figure(figsize=(15, 6))
        plt.plot(plot_times, errors[:valid_length], label='Error')
        plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Prediction Error Over Time')
        plt.xlabel('Time')
        plt.ylabel('Error')
        plt.legend()
        plt.savefig(f"{output_dir}/error_timeline.png")
        plt.close()

        # 4. 誤差の分布
        plt.figure(figsize=(10, 6))
        sns.histplot(errors[:valid_length], bins=30, kde=True)
        plt.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
        plt.title('Error Distribution')
        plt.xlabel('Error')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(f"{output_dir}/error_distribution.png")
        plt.close()

        # 5. 予測vs実測のプロット
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))

        plot_length = min(len(plot_times), len(y_true_original))

        labels = ['Forward Component', 'Lateral Component', 'Magnitude']
        for i in range(3):
            axes[i].plot(plot_times[:plot_length], y_true_original[:plot_length, i], label=f'Actual {labels[i]}')
            axes[i].plot(plot_times[:plot_length], y_pred_original[:plot_length, i], label=f'Predicted {labels[i]}')
            axes[i].set_title(f'{labels[i]}: Actual vs Predicted')
            axes[i].set_xlabel('Time')
            axes[i].set_ylabel(labels[i])
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(f"{output_dir}/prediction_comparison.png")
        plt.close()

        # 6. 相対ベクトルの可視化（軌跡上と原点表示の2つのサブプロット）
        fig = plt.figure(figsize=(24, 12))
        gs = fig.add_gridspec(1, 41)  # 41列のグリッドを作成
        
        # 左側のプロット（軌跡上のベクトル）
        ax1 = fig.add_subplot(gs[0, :19])
        # 中央のプロット（原点からのベクトル）
        ax2 = fig.add_subplot(gs[0, 21:40])
        
        # エラーの正規化（0-1のスケールに）
        normalized_errors = (errors[:valid_length] - errors[:valid_length].min()) / \
                        (errors[:valid_length].max() - errors[:valid_length].min())
        
        # カラーマップの作成
        cmap = plt.cm.RdYlBu_r  # 青から黄色のグラデーション
        
        # プロットの設定
        for ax in [ax1, ax2]:
            ax.set_aspect('equal')
            ax.grid(True)
        
        # ベクトルのスケーリング係数（矢印の長さ調整用）
        scale_factor = 0.1
        scale_factor_origin = 0.5  # 原点表示用の拡大係数
        
        # 軌跡上のベクトル表示（左側のプロット）
        for i in range(valid_length):
            x, y = plot_coords[i]
            forward_comp = plot_data[i, 0]
            lateral_comp = plot_data[i, 1]
            
            if i > 0:
                prev_x, prev_y = plot_coords[i-1]
                direction = np.array([x - prev_x, y - prev_y])
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    
                    forward_vec = direction * forward_comp * scale_factor
                    lateral_vec = np.array([-direction[1], direction[0]]) * lateral_comp * scale_factor
                    
                    if plot_anomalies[i]:
                        color = 'red'
                    else:
                        color = cmap(normalized_errors[i])
                    
                    ax1.arrow(x, y, forward_vec[0], forward_vec[1],
                            head_width=0.01, head_length=0.015,
                            fc=color, ec=color, alpha=0.7)
                    ax1.arrow(x, y, lateral_vec[0], lateral_vec[1],
                            head_width=0.01, head_length=0.015,
                            fc=color, ec=color, alpha=0.7,
                            linestyle='--')
        
        # 原点からのベクトル表示（中央のプロット）
        max_comp = max(abs(plot_data[:, :2]).max(), 1e-6)  # 最大成分値（0除算防止）
        for i in range(valid_length):
            if i > 0:
                forward_comp = plot_data[i, 0]
                lateral_comp = plot_data[i, 1]
                
                prev_x, prev_y = plot_coords[i-1]
                x, y = plot_coords[i]
                direction = np.array([x - prev_x, y - prev_y])
                
                if np.linalg.norm(direction) > 1e-6:
                    direction = direction / np.linalg.norm(direction)
                    
                    # 相対ベクトルを原点から表示
                    if plot_anomalies[i]:
                        color = 'red'
                    else:
                        color = cmap(normalized_errors[i])
                    
                    # 進行方向成分（x軸方向）
                    ax2.arrow(0, 0, 
                            forward_comp / max_comp * scale_factor_origin, 0,
                            head_width=0.02, head_length=0.03,
                            fc=color, ec=color, alpha=0.5)
                    
                    # 横方向成分（y軸方向）
                    ax2.arrow(forward_comp / max_comp * scale_factor_origin, 0,
                            0, lateral_comp / max_comp * scale_factor_origin,
                            head_width=0.02, head_length=0.03,
                            fc=color, ec=color, alpha=0.5,
                            linestyle='--')
                    
                    # 合成ベクトル（点線）
                    ax2.arrow(0, 0,
                            forward_comp / max_comp * scale_factor_origin,
                            lateral_comp / max_comp * scale_factor_origin,
                            head_width=0.02, head_length=0.03,
                            fc='gray', ec='gray', alpha=0.2,
                            linestyle=':')
        
        # 軸の設定
        ax1.set_xlabel('X Coordinate')
        ax1.set_ylabel('Y Coordinate')
        ax1.set_title('Relative Vectors on Trajectory\n(Solid: Forward Component, Dashed: Lateral Component)')
        
        ax2.set_xlabel('Forward Component (Normalized)')
        ax2.set_ylabel('Lateral Component (Normalized)')
        ax2.set_title('Relative Vectors from Origin\n(Solid: Forward, Dashed: Lateral, Dotted: Combined)')
        
        # 原点表示用の軸範囲設定
        ax2.set_xlim(-scale_factor_origin * 1.2, scale_factor_origin * 1.2)
        ax2.set_ylim(-scale_factor_origin * 1.2, scale_factor_origin * 1.2)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # カラーバーの追加（最後の1列を使用）
        cax = fig.add_subplot(gs[0, -1])
        norm = plt.Normalize(errors[:valid_length].min(), errors[:valid_length].max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, cax=cax, label='Prediction Error')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/relative_vectors2.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. 結果のサマリーをテキストファイルに保存
        with open(f"{output_dir}/summary.txt", 'w') as f:
            f.write(f"Velocity-based Anomaly Detection Summary\n")
            f.write(f"====================================\n")
            f.write(f"Total points analyzed: {valid_length}\n")
            f.write(f"Number of anomalies detected: {np.sum(plot_anomalies)}\n")
            f.write(f"Anomaly percentage: {(np.sum(plot_anomalies)/valid_length)*100:.2f}%\n")
            f.write(f"Error threshold: {threshold:.4f}\n")
            f.write(f"Mean error: {np.mean(errors[:valid_length]):.4f}\n")
            f.write(f"Standard deviation of error: {np.std(errors[:valid_length]):.4f}\n")

            f.write("\nVector Statistics:\n")
            f.write(f"Mean forward component: {np.mean(plot_data[:, 0]):.4f}\n")
            f.write(f"Mean lateral component: {np.mean(plot_data[:, 1]):.4f}\n")
            f.write(f"Mean magnitude: {np.mean(plot_data[:, 2]):.4f}\n")
            f.write(f"Max magnitude: {np.max(plot_data[:, 2]):.4f}\n")

            f.write("\nAnomaly Timestamps:\n")
            anomaly_times = plot_times[plot_anomalies]
            for t in anomaly_times:
                f.write(f"Time: {t}\n")

        return output_dir

class CoordinateVisualizer:
    def __init__(self, test_data, test_times, anomalies):
        self.test_data = test_data
        self.test_times = test_times
        self.anomalies = anomalies
        self.trail_length = 15  # 表示する軌跡の長さを調整
        self.highlight_frames = 30  # 異常値を強調表示するフレーム数

    def create_animation(self, output_path='trajectory_animation.mp4', fps=30):
        fig, ax = plt.subplots(figsize=(8, 8))

        # 異常値の履歴を保持する辞書（キーはフレーム番号、値は残りフレーム数）
        anomaly_history = {}

        def init():
            ax.clear()
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.grid(True)
            ax.set_aspect('equal')
            return []

        def update(frame):
            # フレームごとの実際のデータインデックスを計算
            curr_idx = frame * 5
            if curr_idx < self.trail_length or curr_idx >= len(self.test_data):
                return []

            ax.clear()
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            ax.grid(True)
            ax.set_aspect('equal')

            # 表示範囲のデータを取得
            start_idx = max(0, curr_idx - self.trail_length)
            end_idx = curr_idx + 1
            trail_data = self.test_data[start_idx:end_idx]
            trail_anomalies = self.anomalies[start_idx:end_idx]

            # 異常値履歴の更新
            # 新しい異常値の検出と追加
            if trail_anomalies[-1]:
                anomaly_history[curr_idx] = self.highlight_frames

            # 履歴の更新（フレーム数のカウントダウン）
            keys_to_remove = []
            for idx in anomaly_history:
                anomaly_history[idx] -= 1
                if anomaly_history[idx] <= 0:
                    keys_to_remove.append(idx)

            # 期限切れの履歴を削除
            for idx in keys_to_remove:
                del anomaly_history[idx]

            # 軌跡の描画
            points = trail_data.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # 異常値周辺のインデックスを作成
            highlight_indices = set()
            for i in range(len(trail_anomalies)):
                if trail_anomalies[i]:
                    # 異常値の前後の点も含める
                    start_range = max(0, i - 2)
                    end_range = min(len(segments), i + 3)
                    highlight_indices.update(range(start_range, end_range))

            # 軌跡の描画（通常の青とハイライトの赤）
            for i in range(len(segments)):
                if i in highlight_indices:
                    lc = LineCollection([segments[i]], colors=['red'],
                                      linewidth=2, alpha=0.7)
                else:
                    color = plt.cm.Blues(np.linspace(0.2, 1, len(segments))[i])
                    lc = LineCollection([segments[i]], colors=[color],
                                      linewidth=2, alpha=0.7)
                ax.add_collection(lc)

            # 異常値ポイントの表示
            for i in range(len(trail_data)):
                if i < len(trail_anomalies) and trail_anomalies[i]:
                    ax.scatter(trail_data[i, 0], trail_data[i, 1],
                             c='red', s=100, zorder=5)

            # 現在のポイントの表示
            is_current_anomaly = trail_anomalies[-1]
            color = 'red' if is_current_anomaly else 'blue'
            size = 150 if is_current_anomaly else 100
            ax.scatter(trail_data[-1, 0], trail_data[-1, 1],
                      c=color, s=size, zorder=6)

            # 情報表示
            ax.set_title(f'Time: {self.test_times[curr_idx]:.1f}s\n'
                        f'Position: ({trail_data[-1, 0]:.3f}, {trail_data[-1, 1]:.3f})')

            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')

            return []

        # フレーム数を計算
        total_frames = (len(self.test_data) - self.trail_length) // 5

        # アニメーションの作成
        anim = animation.FuncAnimation(fig, update, init_func=init,
                                       frames=total_frames, interval=1000/fps,
                                       blit=True)

        # MP4ファイルとして保存
        writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
        anim.save(output_path, writer=writer)
        plt.close()

        return output_path

if __name__ == "__main__":
    # 入力ファイルのパス（実際のパスに変更してください）
    input_file = 'test_data/video_20240919_164140_r0_c0_tracking_data.csv'
    
    print("=== 異常検知プログラムを開始します ===")
    
    # 1. 検出器の初期化
    print("\n1. 検出器を初期化しています...")
    window_size = 1
    detector = CoordinateAnomalyDetector(window_size=window_size)
    
    # 2. データの読み込みと処理
    print("\n2. データを読み込んでいます...")
    train_data, test_data, test_coords, test_times, full_df = detector.load_and_process_data(input_file)
    print(f"  学習データ数: {len(train_data)}")
    print(f"  テストデータ数: {len(test_data)}")
    
    # 3. モデルの学習
    print("\n3. モデルの学習を開始します...")
    detector.fit(train_data, epochs=10)
    
    # 4. 異常値の検知
    print("\n4. 異常値を検出しています...")
    anomalies, errors, threshold, y_true_original, y_pred_original, errors_by_vector = \
        detector.detect_anomalies(test_data)
    
    # 5. 結果の可視化と保存
    print("\n5. 結果を可視化しています...")
    output_dir = detector.visualize_results(
        test_coords, test_data, test_times, anomalies, errors, threshold,
        y_true_original, y_pred_original, errors_by_vector
    )
    
    # 6. 学習データの保存
    print("\n6. 学習データを保存しています...")
    detector.save_train_data(train_data, output_dir)
    
    # 7. 検証結果の保存
    print("\n7. 検証結果を保存しています...")
    detector.save_test_results(test_coords, test_times, test_data, anomalies, output_dir)
    
    # 8. モデルの保存
    print("\n8. モデルを保存しています...")
    detector.save_model(output_dir)
    
    # 9. アニメーションの作成
    print("\n9. アニメーションを作成しています...")
    coord_anomalies = np.zeros(len(test_coords), dtype=bool)
    valid_length = min(len(coord_anomalies) - window_size, len(anomalies))
    coord_anomalies[window_size:window_size + valid_length] = anomalies[:valid_length]
    
    visualizer = CoordinateVisualizer(
        test_data=test_coords,
        test_times=test_times,
        anomalies=coord_anomalies,
    )
    
    animation_path = os.path.join(output_dir, 'trajectory_animation.mp4')
    # visualizer.create_animation(output_path=animation_path, fps=30)
    
    # 10. 最終結果の表示
    print("\n=== 処理が完了しました ===")
    print(f"\n保存した結果の概要:")
    print(f"  出力ディレクトリ: {output_dir}")
    print(f"  検出された異常値の数: {np.sum(anomalies[:valid_length])}")
    print(f"  異常値判定の閾値: {threshold:.4f}")
    print(f"\n保存されたファイル:")
    print(f"  - 学習データ: {os.path.join(output_dir, 'train_data.csv')}")
    print(f"  - 検証結果: {os.path.join(output_dir, 'test_results.csv')}")
    print(f"  - モデル: {os.path.join(output_dir, 'model')}")
    print(f"  - スケーラー: {os.path.join(output_dir, 'scaler.joblib')}")
    print(f"  - パラメータ: {os.path.join(output_dir, 'parameters.json')}")
    # print(f"  - アニメーション: {animation_path}")
    print(f"  - 可視化結果: {output_dir}/*.png")
