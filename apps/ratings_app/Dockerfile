FROM tiangolo/uwsgi-nginx-flask:python3.6

COPY firefox.conf /etc/nginx/conf.d/
COPY requirements.txt /tmp/

RUN pip3 -q install --user -r /tmp/requirements.txt

COPY ./app /app

ENV NGINX_WORKER_PROCESSES auto