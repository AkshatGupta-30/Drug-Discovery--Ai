FROM ubuntu:18.04

RUN apt-get update

RUN apt-get install -y mysql-server-5.7 python3-mysqldb python3-pip && pip3 install mysqlclient

COPY mysql57_start.sh / 

RUN chmod +x mysql57_start.sh

ENTRYPOINT ["/mysql57_start.sh"]