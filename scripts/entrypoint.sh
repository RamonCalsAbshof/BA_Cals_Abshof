#!/bin/sh

set -e

python manage.py collectstatic --noinput
cp -r parameter_handler/algorithms/* parameter_handler/algorithms_on_host


uwsgi --socket :8000 --master --enable-threads -b 32768 --module backend.wsgi