#!/bin/bash

docker-compose up -d

sleep 10

aws --endpoint-url=http://localhost:4566 s3 mb s3://nyc-duration

python integration_test.py

docker-compose down