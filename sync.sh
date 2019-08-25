#!/bin/bash

rsync -av --exclude-from=sync_ignore.txt ../Canvas-Data-Viz-DEV/ ../Canvas-Data-Viz/
rsync -av ../Canvas-Data-Viz/ dc273@netvisdev.fuqua.duke.edu:~/Canvas-Data-Viz/

ssh dc273@netvisdev.fuqua.duke.edu <<ENDSSH
cd ~/Canvas-Data-Viz
bash rebuild_$1.sh
ENDSSH