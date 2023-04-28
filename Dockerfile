ARG BASE_IMAGE=python:3.9-slim
FROM --platform=linux/amd64 $BASE_IMAGE as runtime-environment

# install project requirements
COPY src/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# add kedro user
ARG KEDRO_UID=999
ARG KEDRO_GID=0
RUN groupadd -f -g ${KEDRO_GID} kedro_group && \
    useradd -m -d /home/kedro_docker -s /bin/bash -g ${KEDRO_GID} -u ${KEDRO_UID} kedro_docker

WORKDIR /home/kedro_docker
USER kedro_docker

FROM runtime-environment

# copy the whole project except what is in .dockerignore
ARG KEDRO_UID=999
ARG KEDRO_GID=0
COPY --chown=${KEDRO_UID}:${KEDRO_GID} . .
RUN pip install .

EXPOSE 5001

#CMD ["gunicorn", "--bind", ":5001", "api:app"]
CMD ["python", "src/airline_passenger_satisfaction/app/api.py"]
