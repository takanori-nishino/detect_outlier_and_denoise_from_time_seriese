import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
import argparse

UNIT_TO_METER = 0.3  # 1 unit = 0.3 meters

def calculate_vector_magnitude(df: pd.DataFrame) -> np.ndarray:
    """フィルタ済み座標からベクトルの大きさを計算（外れ値を除外）"""
    # 外れ値でないデータのみを使用
    valid_data = df[~df['is_anomaly']].copy()
    
    # 座標の差分を計算
    dx = np.diff(valid_data['filtered_x'])
    dy = np.diff(valid_data['filtered_y'])
    
    # ベクトルの大きさを計算（単位時間あたり）
    magnitudes = np.sqrt(dx**2 + dy**2)
    
    # 時間差分を計算
    dt = np.diff(valid_data['time'])
    
    # 単位時間（1秒）あたりのベクトル量に正規化してメートルに変換し、時間単位を時間に変換
    normalized_magnitudes = magnitudes / dt * UNIT_TO_METER * 3600  # m/s から m/hour に変換
    
    return normalized_magnitudes, valid_data['time'].values[:-1]

def analyze_movement(file_path: str, start_time: datetime, category: str, sample_name: str) -> pd.DataFrame:
    """フィルタ済み座標データを分析（外れ値を除外）"""
    try:
        # データの読み込み
        df = pd.read_csv(file_path)
        
        # 1時間後から25時間後までのデータを使用
        start_sec = 3600  # 1時間
        end_sec = 25 * 3600  # 25時間
        df = df[(df['time'] >= start_sec) & (df['time'] < end_sec)]
        
        # ベクトル量の計算（外れ値を除外）
        vector_magnitudes, times = calculate_vector_magnitude(df)
        
        # 実際の時刻を計算（開始時刻からの経過時間を加算）
        actual_hours = [(start_time + timedelta(seconds=t)).hour for t in times]
        
        # 結果を格納するデータフレーム
        results = pd.DataFrame({
            'hour': actual_hours,  # 実際の時刻（時）を使用
            'vector_magnitude': vector_magnitudes
        })
        
        # 時間ごとの平均を計算
        hourly_means = results.groupby('hour')['vector_magnitude'].mean().reset_index()
        hourly_means['category'] = category
        hourly_means['sample_name'] = str(sample_name)
        
        return hourly_means
    
    except Exception as e:
        print(f"Error analyzing {file_path}: {str(e)}")
        return None

def plot_movement_by_category(all_results: pd.DataFrame, output_dir: str):
    """カテゴリごとに移動量の時間変化をプロット"""
    plt.figure(figsize=(12, 8))
    
    colors = [
        '#1f77b4',  # 青
        '#d62728',  # 赤
        '#2ca02c',  # 緑
        '#9467bd',  # 紫
        '#8c564b',  # 茶
        '#e377c2',  # ピンク
        '#7f7f7f',  # グレー
        '#bcbd22',  # 黄緑
        '#17becf',  # 水色
        '#ff7f0e',  # オレンジ
    ]
    
    categories = all_results['category'].unique()
    color_dict = {category: colors[i % len(colors)] for i, category in enumerate(categories)}
    
    samples = all_results['sample_name'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    marker_dict = {str(sample): markers[i % len(markers)] for i, sample in enumerate(samples)}
    
    for category in categories:
        category_data = all_results[all_results['category'] == category]
        for sample in category_data['sample_name'].unique():
            data = all_results[
                (all_results['category'] == category) & 
                (all_results['sample_name'] == sample)
            ].sort_values('hour')
            
            # 24時間周期でデータを循環させる
            cycled_data = data.copy()
            cycled_data['hour'] = cycled_data['hour'] % 24
            cycled_data = cycled_data.sort_values('hour')
            
            plt.plot(cycled_data['hour'], cycled_data['vector_magnitude'],
                    marker=marker_dict[sample], 
                    color=color_dict[category],
                    label=f'{category} - {sample}',
                    markersize=8,
                    linewidth=2)
    
    plt.xlabel('Time of Day (Hour)')
    plt.ylabel('Hourly Movement (m)')
    plt.title('Hourly Movement Over Time\n(Anomalies Excluded)')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # x軸の設定（0-23時）
    plt.xticks(range(0, 24, 2))
    plt.xlim(0, 23)
    
    # y軸を0から開始
    plt.ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'movement_analysis_by_category.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def process_filtered_data(metadata_file: str, output_dir: str):
    """メタデータに基づいてフィルタ済みの座標データを処理"""
    metadata = pd.read_csv(metadata_file, dtype={'sample_name': str})
    all_results = []
    
    for _, row in metadata.iterrows():
        filtered_coords_path = os.path.join(os.path.dirname(metadata_file), row['filename'])
        
        if os.path.exists(filtered_coords_path):
            print(f"Processing {filtered_coords_path}")
            
            start_time = datetime.strptime(row['start_time'], '%H:%M')
            
            hourly_data = analyze_movement(
                filtered_coords_path,
                start_time,
                row['category'],
                row['sample_name']
            )
            
            if hourly_data is not None:
                all_results.append(hourly_data)
        else:
            print(f"File not found: {filtered_coords_path}")
    
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        
        combined_results.to_csv(os.path.join(output_dir, 'hourly_movement_data.csv'), index=False)
        
        plot_movement_by_category(combined_results, output_dir)
        return combined_results
    else:
        print("No files were processed successfully")
        return None

def main():
    parser = argparse.ArgumentParser(description='Analyze movement data from filtered coordinates')
    parser.add_argument('metadata', help='Path to the metadata CSV file')
    parser.add_argument('-o', '--output', default='results',
                        help='Output directory (default: results)')
    
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    results = process_filtered_data(args.metadata, args.output)
    
    if results is not None:
        print(f"\nResults have been saved to {args.output}")
        print(f"- Graph: {os.path.join(args.output, 'movement_analysis_by_category.png')}")
        print(f"- Data: {os.path.join(args.output, 'hourly_movement_data.csv')}")

if __name__ == "__main__":
    main()