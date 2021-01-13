export MYSQL_USER=
export MYSQL_PASSWORD=
export MYSQL_IP=
export MYSQL_DB=

python3 manage.py makemigrations
python3 manage.py migrate
gunicorn backend.wsgi:application --bind 0.0.0.0:8000
