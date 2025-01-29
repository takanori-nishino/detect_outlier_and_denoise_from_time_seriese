import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os

def create_trajectory_animation(csv_path, output_path='trajectory_animation.mp4', fps=10):
    """
    軌跡アニメーションを作成する関数
    """
    # CSVファイルの読み込み
    df = pd.read_csv(csv_path)
    
    # データの内容を確認
    print("Data info:")
    print(f"Time range: {df['time'].min():.1f} to {df['time'].max():.1f} seconds")
    print(f"Number of data points: {len(df)}")
    
    # 時間を秒単位で扱う
    window_size_sec = 10  # 表示する時間幅（秒）
    slide_size_sec = 5    # スライドさせる時間幅（秒）
    
    # アニメーションの設定
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # x軸とy軸の範囲を計算
    x_margin = (df['filtered_x'].max() - df['filtered_x'].min()) * 0.1
    y_margin = (df['filtered_y'].max() - df['filtered_y'].min()) * 0.1
    x_min = df['filtered_x'].min() - x_margin
    x_max = df['filtered_x'].max() + x_margin
    y_min = df['filtered_y'].min() - y_margin
    y_max = df['filtered_y'].max() + y_margin
    
    print(f"Coordinate ranges: X [{x_min:.2f}, {x_max:.2f}], Y [{y_min:.2f}, {y_max:.2f}]")
    
    def update(frame):
        ax.clear()
        
        # 現在のフレームの時間範囲を計算
        start_time = frame * slide_size_sec
        end_time = start_time + window_size_sec
        
        # 時間範囲内のデータを抽出
        mask = (df['time'] >= start_time) & (df['time'] < end_time)
        current_data = df[mask].copy()
        
        if len(current_data) > 0:
            print(f"Frame {frame}: {len(current_data)} points, time {start_time:.1f}-{end_time:.1f}s")
            
            # インデックスをリセット
            current_data = current_data.reset_index()
            
            # 軌跡を分割してプロット（外れ値の前後で色を変える）
            for i in range(len(current_data) - 1):
                is_anomaly = (current_data['is_anomaly'].iloc[i] or 
                            current_data['is_anomaly'].iloc[i + 1])
                
                color = 'red' if is_anomaly else 'blue'
                alpha = 0.8
                
                ax.plot(current_data['filtered_x'].iloc[i:i+2],
                       current_data['filtered_y'].iloc[i:i+2],
                       color=color, linewidth=1, alpha=alpha)
            
            # 点をプロット（外れ値は赤、通常は青）
            normal_points = ~current_data['is_anomaly']
            anomaly_points = current_data['is_anomaly']
            
            # 通常の点
            ax.plot(current_data.loc[normal_points, 'filtered_x'],
                   current_data.loc[normal_points, 'filtered_y'],
                   'bo', markersize=6, alpha=0.6)
            
            # 異常点
            ax.plot(current_data.loc[anomaly_points, 'filtered_x'],
                   current_data.loc[anomaly_points, 'filtered_y'],
                   'ro', markersize=8, alpha=0.8)
            
            # 現在位置を大きな点で表示
            current_color = 'red' if current_data['is_anomaly'].iloc[-1] else 'blue'
            ax.plot(current_data['filtered_x'].iloc[-1],
                   current_data['filtered_y'].iloc[-1],
                   'o', color=current_color, markersize=10)
        
        # グラフの設定
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f'Time Window: {start_time:.1f}-{end_time:.1f} s')
        ax.set_xlabel('Normalized X Position')
        ax.set_ylabel('Normalized Y Position')
        
        return ax.get_children()
    
    # 総フレーム数を計算
    total_duration = df['time'].max() - df['time'].min()
    n_frames = int((total_duration - window_size_sec) / slide_size_sec) + 1
    print(f"Calculated {n_frames} frames for {total_duration:.1f} seconds of data")
    
    # アニメーションの作成
    ani = animation.FuncAnimation(fig, update, frames=n_frames,
                                interval=1000/fps, blit=True)
    
    # 動画として保存
    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    ani.save(output_path, writer=writer)
    plt.close()
    
    print(f"\nAnimation saved to: {output_path}")
    print(f"Total frames: {n_frames}")
    print(f"Duration: {n_frames/fps:.1f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Create trajectory animation from filtered coordinate data')
    parser.add_argument('input_csv', type=str, help='Input CSV file with filtered coordinates')
    parser.add_argument('--output', type=str, default='trajectory_animation.mp4',
                        help='Output video file path [default: trajectory_animation.mp4]')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for the output video [default: 30]')
    parser.add_argument('--window', type=float, default=10.0,
                        help='Time window size in seconds [default: 10.0]')
    parser.add_argument('--slide', type=float, default=5.0,
                        help='Time window slide in seconds [default: 5.0]')
    
    args = parser.parse_args()
    
    # 出力ディレクトリがディレクトリとして指定された場合の処理
    if os.path.isdir(args.output):
        output_path = os.path.join(args.output, 'trajectory_animation.mp4')
    else:
        output_path = args.output
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    # 拡張子の確認と追加
    if not output_path.lower().endswith('.mp4'):
        output_path += '.mp4'
    
    # アニメーションの作成
    create_trajectory_animation(args.input_csv, output_path, args.fps)

if __name__ == "__main__":
    main()