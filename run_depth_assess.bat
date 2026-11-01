@echo off
cd /d "%~dp0"
if not exist logs mkdir logs

set PY=.venv\Scripts\python.exe
set RUN=improvedQuant\v1_classifiers\run.py
set COMMON=-i data/ts -p data/indices -d improvedQuant/dataSetLists/ucr142.txt -r 30 -v 4

echo %date% %time% start >> logs\progress.log

%PY% %RUN% %COMMON% --depths 0,2,3,4,5 -o results/no_depth_1.csv > logs\no_depth_1.log 2>&1
echo %date% %time% no_depth_1 done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 0,1,3,4,5 -o results/no_depth_2.csv > logs\no_depth_2.log 2>&1
echo %date% %time% no_depth_2 done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 0,1,2,4,5 -o results/no_depth_3.csv > logs\no_depth_3.log 2>&1
echo %date% %time% no_depth_3 done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 0,1,2,3,5 -o results/no_depth_4.csv > logs\no_depth_4.log 2>&1
echo %date% %time% no_depth_4 done >> logs\progress.log

%PY% %RUN% %COMMON% --depths 0,1,2 -o results/no_depth_3_4_5.csv > logs\no_depth_3_4_5.log 2>&1
echo %date% %time% no_depth_3_4_5 done >> logs\progress.log