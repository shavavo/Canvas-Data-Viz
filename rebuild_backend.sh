#!/bin/bash

pushd /home/FUQUA/dc273/Canvas-Data-Viz/
bash update_sql_models.sh
popd

pushd /home/FUQUA/dc273/Canvas-Data-Viz/backend/
kill -9 `cat save_pid.txt`
nohup python3 -u Canvas_to_SQL.py 2>&1 &
echo $! > save_pid.txt
popd