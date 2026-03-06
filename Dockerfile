FROM python:3.12-slim

WORKDIR /app

COPY shin/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shin/ ./shin/
COPY sysprompt.txt /app/sysprompt.txt

ENV HOST=0.0.0.0
ENV PORT=4000
ENV GATEWAY_SYSTEM_PROMPT_FILE=/app/sysprompt.txt

EXPOSE 4000

CMD ["python", "-m", "shin.run"]
