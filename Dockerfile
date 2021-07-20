FROM python:3.9

RUN useradd -m  abhay && apt-get update 

WORKDIR /home/abhay/app

COPY . /home/abhay/app

RUN chown abhay:abhay -R /home/abhay/app

RUN pip install --upgrade pip setuptools && pip install -r requirements.txt

USER abhay

CMD ["sh","run.sh"] 

