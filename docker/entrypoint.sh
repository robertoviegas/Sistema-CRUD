#!/usr/bin/env bash
set -euo pipefail

# Default environment
: "${APP_ENV:=dev}"
: "${DB_URL:=sqlite+pysqlite:///./crud.db}"

echo "[entrypoint] Waiting for database if needed..."
if [[ "$DB_URL" == postgresql* ]] || [[ "$DB_URL" == postgres* ]]; then
  # wait for Postgres
  python - <<'PY'
import os, time
import psycopg
from urllib.parse import urlparse

db_url = os.environ.get('DB_URL')
u = urlparse(db_url.replace('postgresql+psycopg', 'postgresql'))
host = u.hostname or 'db'
port = u.port or 5432
for i in range(60):
    try:
        with psycopg.connect(f"host={host} port={port} user={u.username} password={u.password} dbname={u.path.lstrip('/')}", connect_timeout=2):
            print('[entrypoint] Database is up')
            break
    except Exception as e:
        print('[entrypoint] Waiting DB...', e)
        time.sleep(2)
PY
fi

echo "[entrypoint] Running migrations/init-db..."
python manage.py init-db || true

echo "[entrypoint] Starting API..."
exec python manage.py run --host 0.0.0.0 --port 8000




