export MYSQL_USER=
export MYSQL_PASSWORD=
export MYSQL_IP=
export MYSQL_DB=

python3 manage.py makemigrations
python3 manage.py migrate
python3 manage.py runserver
