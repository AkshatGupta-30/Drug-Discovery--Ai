#!/bin/bash

/etc/init.d/mysql start
sleep 10

mysql --user=root <<EOF
ALTER USER 'root'@'localhost' IDENTIFIED BY 'root';
DELETE FROM mysql.user WHERE User='';
DELETE FROM mysql.user WHERE User='root' AND Host NOT IN ('localhost','127.0.0.1', '::1');
DROP DATABASE IF EXISTS test;
DELETE FROM mysql.db WHERE Db='test' OR Db='test\\_&';
CREATE USER 'mlflow_user'@'localhost' IDENTIFIED BY 'mlflow';
GRANT ALL ON db_mlflow.* TO 'mlflow_user'@'localhost';
FLUSH PRIVILEGES;
CREATE DATABASE IF NOT EXISTS db_mlflow;
SHOW DATABASES;
EOF

tail -f /var/log/mysql/error.log