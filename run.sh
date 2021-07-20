gunicorn -w 2 -t 10 --bind 0.0.0.0:5000 wsgi:app 
