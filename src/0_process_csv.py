#!/usr/bin/env python3
import argparse
import glob
import os
import sys
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import logging
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from detect_outlier import CoordinateAnomalyDetector, CoordinateVisualizer

def apply_lowpass_filter(data: np.ndarray, normalized_cutoff: float, order: int = 4) -> np.ndarray:
    """
    ローパスフィルタを適用
    Args:
        data: 入力データ
        normalized_cutoff: 正規化されたカットオフ周波数 (0 < freq < 1)
        order: フィルタの次数
    """
    if not 0 < normalized_cutoff < 1:
        raise ValueError("Normalized cutoff frequency must be between 0 and 1")
        
    b, a = butter(order, normalized_cutoff, btype='low')
    return filtfilt(b, a, data, axis=0)

def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """ロギングの設定"""
    logger = logging.getLogger('AnomalyDetection')
    
    # すでにハンドラーが設定されている場合は、それらを削除
    if logger.handlers:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
    
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def process_single_file(input_file: str, output_dir: str, window_size: int,
                       threshold_sigma: float, epochs: int, cutoff_freq: float,
                       filter_order: int, log_file: Optional[str] = None) -> bool:
    """単一ファイルの処理"""
    try:
        # 各プロセスで独自のloggerを作成
        logger = setup_logger(log_file)
        logger.info(f"Started processing {input_file}")
        
        # 出力ディレクトリの作成
        file_basename = os.path.splitext(os.path.basename(input_file))[0]
        file_output_dir = os.path.join(
            output_dir,
            f"{file_basename}_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        os.makedirs(file_output_dir, exist_ok=True)
        
        # 以下の各ステップで例外が発生した場合の詳細なログ
        try:
            # 異常検出器の初期化と実行
            logger.info(f"Initializing detector for {input_file}")
            detector = CoordinateAnomalyDetector(
                window_size=window_size,
                threshold_sigma=threshold_sigma
            )
            
            # データの読み込みと処理
            logger.info(f"Loading and processing data from {input_file}")
            train_data, test_data, test_coords, test_times, full_df = \
                detector.load_and_process_data(input_file)
            
            # モデルの学習
            logger.info(f"Training model")
            detector.fit(train_data, epochs=epochs)
            
            # 異常値の検出
            logger.info(f"Detecting anomalies")
            anomalies, errors, threshold, y_true_original, y_pred_original, errors_by_vector = \
                detector.detect_anomalies(test_data)
            
            # 結果の保存
            logger.info(f"Saving results")
            detector.save_train_data(train_data, file_output_dir)
            detector.save_test_results(test_coords, test_times, test_data, 
                                     anomalies, file_output_dir)
            detector.save_model(file_output_dir)
            
            # ローパスフィルタの適用
            logger.info(f"Applying lowpass filter")
            filtered_coords = np.zeros_like(test_coords)
            filtered_coords[:, 0] = apply_lowpass_filter(test_coords[:, 0], cutoff_freq)
            filtered_coords[:, 1] = apply_lowpass_filter(test_coords[:, 1], cutoff_freq)

            # データの長さを確認
            logger.info(f"Verifying data lengths: times={len(test_times)}, coords={len(test_coords)}, anomalies={len(anomalies)}")

            # anomaliesの長さを調整（必要な場合）
            if len(anomalies) < len(test_times):
                # 不足分をFalseで埋める
                padding = np.zeros(len(test_times) - len(anomalies), dtype=bool)
                anomalies = np.concatenate([anomalies, padding])
            elif len(anomalies) > len(test_times):
                # 余分を切り捨てる
                anomalies = anomalies[:len(test_times)]

            # フィルタ結果をCSVとして保存
            filtered_df = pd.DataFrame({
                'time': test_times,
                'x': test_coords[:, 0],
                'y': test_coords[:, 1],
                'filtered_x': filtered_coords[:, 0],
                'filtered_y': filtered_coords[:, 1],
                'is_anomaly': anomalies
            })

            logger.info(f"Created DataFrame with {len(filtered_df)} rows")

            # フィルタ結果をCSVとして保存する行を追加
            filtered_csv_path = os.path.join(file_output_dir, 'filtered_coordinates.csv')
            filtered_df.to_csv(filtered_csv_path, index=False)
            logger.info(f"Saved filtered coordinates to {filtered_csv_path}")

            logger.info(f"Successfully processed {input_file}")
            return True
                            
        except Exception as e:
            logger.error(f"Error in processing step for {input_file}: {str(e)}")
            logger.error(f"Error traceback: ", exc_info=True)
            return False
            
    except Exception as e:
        print(f"Critical error in process_single_file for {input_file}: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description='Batch process coordinate data for anomaly detection and filtering'
    )
    parser.add_argument(
        'input_pattern',
        help='Input file pattern (e.g., "data/*.csv" or specific file)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='analysis_results',
        help='Output directory for results (default: analysis_results)'
    )
    parser.add_argument(
        '-w', '--window-size',
        type=int,
        default=10,
        help='Window size for anomaly detection (default: 10)'
    )
    parser.add_argument(
        '-t', '--threshold-sigma',
        type=float,
        default=3.0,
        help='Threshold sigma for anomaly detection (default: 3.0)'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '-f', '--filter-cutoff',
        type=float,
        default=0.1,  # 0から1の間の正規化された周波数
        help='Normalized cutoff frequency (0 to 1, default: 0.1)'
    )
    parser.add_argument(
        '--filter-order',
        type=int,
        default=4,
        help='Order of the lowpass filter (default: 4)'
    )
    parser.add_argument(
        '--log-file',
        help='Log file path (optional)'
    )
    
    args = parser.parse_args()
    
    # メインプロセスのロガー設定
    logger = setup_logger(args.log_file)
    
    # 入力ファイルの検索
    input_files = glob.glob(args.input_pattern)
    if not input_files:
        logger.error(f"No files found matching pattern: {args.input_pattern}")
        return 1
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Found {len(input_files)} files to process")
    logger.info(f"Output directory: {args.output_dir}")
    
    # 処理の実行
    logger.info("Processing files sequentially")
    success_count = sum(
        process_single_file(
            input_file,
            args.output_dir,
            args.window_size,
            args.threshold_sigma,
            args.epochs,
            args.filter_cutoff,
            args.filter_order,
            args.log_file
        )
        for input_file in input_files
    )
    
    # 処理結果の表示
    total_files = len(input_files)
    logger.info(f"\nProcessing completed:")
    logger.info(f"Successfully processed: {success_count}/{total_files} files")
    logger.info(f"Failed: {total_files - success_count} files")
    
    return 0 if success_count == total_files else 1

if __name__ == "__main__":
    sys.exit(main())

# --filterorderはローパスフィルタで使うカットオフの次数（シャープネス）。2次か4次を使うことが多い。
# f はノーマライズされた周波数のカットオフ
# python src/0_process_csv.py "data/*.csv" -o results -w 1 -t 3.0 -e 10 -f 0.2 --filter-order 4
# python src/0_process_csv.py "data/*.csv" -o results2 -w 1 -t 4.0 -e 3 -f 0.2 --filter-order 4