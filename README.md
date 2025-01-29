# detect_outlier_and_denoise_from_time_seriese
This program detect outlier and denoise positional data from time series

## Install
```
git clone https://github.com/takanori-nishino/detect_outlier_and_denoise_from_time_seriese.git
cd detect_outlier_and_denoise_from_time_seriese
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# Usage
```
# --filterorderはローパスフィルタで使うカットオフの次数（シャープネス）。2次か4次を使うことが多い。
# f はノーマライズされた周波数のカットオフ
python src/0_process_csv.py "data/*.csv" -o results -w 1 -t 3.0 -e 10 -f 0.2 --filter-order 4
```