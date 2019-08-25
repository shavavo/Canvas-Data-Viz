#!/bin/bash

bash update_sql_models.sh
docker stop ratings
docker rm ratings
(cd apps/ratings_app/ && docker build -t ratings .)
docker run -d --name ratings -p 127.0.0.1:81:80 ratings