#!/bin/bash

bash update_sql_models.sh
docker stop analytics
docker rm analytics
(cd apps/analytics_app/ && docker build -t analytics .)
docker run -d --name analytics -p 127.0.0.1:82:80 analytics