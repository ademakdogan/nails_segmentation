#-----------------------
FROM python:3.8-slim-buster
#FROM --platform=linux/x86_64 python:3.8
#FROM --platform=linux/x86_64 python:3.8
RUN apt-get update && apt-get install make
RUN apt-get install gcc -y
WORKDIR /opt/DeepLab
COPY . .
RUN pip3 install --upgrade pip
RUN make install
ENTRYPOINT ["python","src/main.py"]