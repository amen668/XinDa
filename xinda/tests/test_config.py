"""Database settings must not ship a usable password or override user configuration."""
from urllib.parse import urlsplit

from xinda.config import Settings


def test_database_default_does_not_include_password(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    monkeypatch.delenv('database_url', raising=False)
    settings = Settings(_env_file=None)
    # Inspect only the parsed password's presence; never print its value on failure.
    assert urlsplit(settings.database_url).password is None, 'Embedded database password'


def test_database_connection_can_be_configured(monkeypatch):
    url = 'postgresql+asyncpg://test_user@example.invalid:5432/test_db'
    monkeypatch.setenv('DATABASE_URL', url)
    assert Settings(_env_file=None).database_url == url
