FROM nginx

# VOLUME /Users/marcelloqualitade/Documents/docker_tests/cenicana_app
# WORKDIR /Users/marcelloqualitade/Documents/docker_tests/cenicana_app

COPY ./requirements.txt /requirements.txt
# COPY ./run.sh /run.sh

RUN apt-get update
RUN apt-get install -y python3 python3-pip
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8899/tcp

COPY . .

# ENTRYPOINT ["/run.sh"]