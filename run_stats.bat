@echo off
cd /d "%~dp0"
if not exist logs mkdir logs

set PY=.venv\Scripts\python.exe
set RUN=improvedQuant\basic\run.py
set COMMON=-i data/ts -p data/indices -d improvedQuant/dataSetLists/ucr142.txt -r 30 -v 4
set STATS_ALL=mean std min max nmp skew kurt slope argmax argmin mac mcr

echo %date% %time% start >> logs\progress.log

%PY% %RUN% %COMMON% --interval_features quantile -o results/basic_baseline.csv > logs\basic_baseline.log 2>&1
echo %date% %time% baseline done >> logs\progress.log

%PY% %RUN% %COMMON% --interval_features stats --stats %STATS_ALL% -o results/all_stats.csv > logs\all_stats.log 2>&1
echo %date% %time% all_stats done >> logs\progress.log

%PY% %RUN% %COMMON% --interval_features auto --stats_ratio 2 --stats %STATS_ALL% -o results/half_stats.csv > logs\half_stats.log 2>&1
echo %date% %time% half_stats done >> logs\progress.log

%PY% %RUN% %COMMON% --interval_features auto --stats_ratio 1 --stats %STATS_ALL% -o results/equal_stats.csv > logs\equal_stats.log 2>&1
echo %date% %time% equal_stats done >> logs\progress.log

%PY% %RUN% %COMMON% --interval_features auto --stats_ratio 0.5 --stats %STATS_ALL% -o results/double_stats.csv > logs\double_stats.log 2>&1
echo %date% %time% double_stats done >> logs\progress.log