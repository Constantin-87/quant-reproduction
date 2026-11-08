@echo off
cd /d "%~dp0"
if not exist logs mkdir logs

set PY=.venv\Scripts\python.exe
set RUN=improvedQuant\v1_classifiers\run.py
set COMMON=-i data/ts -p data/indices -d improvedQuant/dataSetLists/ucr142.txt -r 30 -v 4

echo %date% %time% start >> logs\progress.log

%PY% %RUN% %COMMON% -o results/depths_0_5_baseline.csv > logs\depths_0_5_baseline.log 2>&1
echo %date% %time% baseline done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 0,1,2,3,4 -o results/no_depth_5.csv > logs\no_depth_5.log 2>&1
echo %date% %time% no_depth_5 done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 0,1,2,3 -o results/no_depth_4_5.csv > logs\no_depth_4_5.log 2>&1
echo %date% %time% no_depth_4_5 done >> logs\progress.log

%PY% %RUN% %COMMON% --interval_types alternating -o results/alternating_depths.csv > logs\alternating_depths.log 2>&1
echo %date% %time% alternating done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 0,2,4 -o results/even_depths.csv > logs\even_depths.log 2>&1
echo %date% %time% even done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 1,3,5 -o results/odd_depths.csv > logs\odd_depths.log 2>&1
echo %date% %time% odd done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 1,2,3,4,5 -o results/no_depth_0.csv > logs\no_depth_0.log 2>&1
echo %date% %time% no_depth_0 done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 2,3,4,5 -o results/no_depth_0_1.csv > logs\no_depth_0_1.log 2>&1
echo %date% %time% no_depth_0_1 done >> logs\progress.log