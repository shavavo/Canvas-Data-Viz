#base image with nginx web server, uWGSI app server, and Flask - allows you to deploy Flask app with Nginx as a reverse proxy. 
FROM tiangolo/uwsgi-nginx-flask:python3.6
#local host to container filesystem - firefox.conf from local host directory to /etc/nginx/conf.d/ directory of the Docker container - used to configure Nginx with Firefox specific settings
COPY firefox.conf /etc/nginx/conf.d/
#local host's requirements.txt file is copied into /tmp/ directory of the docker container
COPY requirements.txt /tmp/
#install python packages inside of docker container
RUN pip3 -q install --user -r /tmp/requirements.txt
#./app of local host directory is copied to /app directory of docker container
COPY ./app /app
#enviornment variable to auto determine optimal number of worker processes based on system resource allocation. 
ENV NGINX_WORKER_PROCESSES auto
