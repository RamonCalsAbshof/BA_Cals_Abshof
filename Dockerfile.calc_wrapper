#Calculation Wrapper
FROM ubuntu:20.04
WORKDIR /usr/src/app
RUN apt update -y && \
apt install -y python3-pip && \
pip3 install --upgrade pip 

RUN pip3 install \  
sklearn \
sqlalchemy \
pymongo \
numpy \
pandas \
scipy \
xgboost \
scikit-learn \
pymysql \
cryptography
