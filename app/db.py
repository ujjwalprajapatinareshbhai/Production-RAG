import psycopg

from app.config import settings


def get_connection():
    """Open a connection to the Production-RAG PostgreSQL database."""
    return psycopg.connect(
        host=settings.postgres_host,
        port=settings.postgres_port,
        dbname=settings.postgres_db,
        user=settings.postgres_user,
        password=settings.postgres_password,
        connect_timeout=5,
    )