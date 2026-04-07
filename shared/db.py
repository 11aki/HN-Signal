"""Shared database connection helper — used by every service."""
import os
import psycopg2
from psycopg2.extras import RealDictCursor


def get_conn():
    """
    Open and return a new Postgres connection.

    Credentials come from environment variables set in docker-compose.yml.
    RealDictCursor makes every row behave like a dict (row["column_name"])
    instead of a tuple (row[0]), which is easier to work with.

    Caller is responsible for closing the connection (use try/finally).
    """
    return psycopg2.connect(
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        cursor_factory=RealDictCursor,
    )
