#!/bin/bash

pushd /home/FUQUA/dc273/Canvas-Data-Viz/
bash rebuild_ratings.sh
bash rebuild_analytics.sh
bash rebuild_predictive.sh
popd