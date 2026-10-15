FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install /usr/local
COPY helpdesk ./helpdesk
COPY scripts ./scripts
COPY app.py .
RUN useradd -m -u 1000 app && mkdir -p /app/data && chown -R app /app
USER app
ENV PORT=8000
EXPOSE 8000
# The API spawns the MCP server as a child process over stdio
CMD ["sh", "-c", "uvicorn helpdesk.api:app --host 0.0.0.0 --port $PORT"]
