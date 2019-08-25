#!/bin/bash

bash update_sql_models.sh
docker stop predictive
docker rm predictive
(cd apps/predictive_app/ && docker build -t predictive .)
docker run -d --name predictive -p 127.0.0.1:83:80 predictive
