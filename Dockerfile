FROM python:3.12-slim
RUN apt-get update \
&& apt-get install -y --no-install-recommends git \
&& apt-get purge -y --auto-remove \
&& rm -rf /var/lib/apt/lists/*
WORKDIR /code
COPY ./backend/requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH
ENV MPLCONFIGDIR=$HOME/temp
WORKDIR $HOME/backend
COPY --chown=user ./backend $HOME/backend
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]