from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Optional
from urllib.parse import quote_plus, urlsplit

import certifi
from pymongo import MongoClient


def _build_mongodb_uri() -> str:
    # Prefer a full URI when provided (Atlas/local, replica sets, etc.).
    uri = os.environ.get("MONGODB_URI")
    if uri:
        return uri

    host = os.environ.get("MONGO_HOST", "localhost")
    port = os.environ.get("MONGO_PORT", "27017")
    db = os.environ.get("MONGO_DB", "guardian")

    user = os.environ.get("MONGO_USER")
    password = os.environ.get("MONGO_PASSWORD")
    auth_source = os.environ.get("MONGO_AUTH_SOURCE", db)

    if user and password:
        u = quote_plus(user)
        p = quote_plus(password)
        a = quote_plus(auth_source)
        return f"mongodb://{u}:{p}@{host}:{port}/{db}?authSource={a}"

    return f"mongodb://{host}:{port}/{db}"

def mongo_uri_summary(uri: Optional[str] = None) -> dict:
    """
    Best-effort, non-sensitive summary of the configured Mongo URI.
    Useful for debugging whether the backend is using Atlas vs localhost.
    """
    u = uri or _build_mongodb_uri()
    parts = urlsplit(u)

    netloc = parts.netloc
    # Redact userinfo if present: user:pass@host -> host
    if "@" in netloc:
        netloc = netloc.split("@", 1)[1]

    db_name = (parts.path or "").lstrip("/") or None

    return {
        "scheme": parts.scheme or None,
        "host": netloc or None,
        "db": db_name,
    }


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    # Keep timeouts short-ish so /health doesn't hang when DB is down,
    # but long enough for Atlas TLS + replica set discovery.
    uri = _build_mongodb_uri()

    server_sel_ms = int(os.environ.get("MONGO_SERVER_SELECTION_TIMEOUT_MS", "5000"))
    connect_ms = int(os.environ.get("MONGO_CONNECT_TIMEOUT_MS", "5000"))

    kwargs = {
        "serverSelectionTimeoutMS": server_sel_ms,
        "connectTimeoutMS": connect_ms,
    }

    # Force a known CA bundle when using TLS (Atlas defaults to TLS).
    if uri.startswith("mongodb+srv://") or "tls=true" in uri or "ssl=true" in uri:
        kwargs["tlsCAFile"] = certifi.where()

    return MongoClient(uri, **kwargs)


def mongo_ping() -> bool:
    client = get_mongo_client()
    client.admin.command("ping")
    return True


def mongo_check() -> tuple[bool, dict, Optional[str]]:
    """
    Returns (ok, summary, error_string).
    """
    summary = mongo_uri_summary()
    try:
        mongo_ping()
        return True, summary, None
    except Exception as e:
        # Avoid returning secrets; exception messages typically do not include creds.
        return False, summary, f"{e.__class__.__name__}: {e}"


def get_database():
    client = get_mongo_client()
    try:
        db = client.get_default_database()
    except Exception:
        db = None
    if db is None:
        db_name = os.environ.get("MONGO_DB", "guardian")
        db = client[db_name]
    return db


def get_senior_users(limit: int = 50) -> list[dict]:
    db = get_database()
    return list(db["users"].find({"role": "senior"}).sort("created_at", -1).limit(limit))


def get_latest_checkins(user_ids: list) -> dict:
    if not user_ids:
        return {}
    db = get_database()
    pipeline = [
        {"$match": {"user_id": {"$in": user_ids}}},
        {"$sort": {"completed_at": -1}},
        {
            "$group": {
                "_id": "$user_id",
                "latest": {"$first": "$$ROOT"}
            }
        }
    ]
    results = db["checkin_history"].aggregate(pipeline)
    latest_by_user = {}
    for item in results:
        latest_by_user[item["_id"]] = item.get("latest")
    return latest_by_user


def get_dashboard_analytics(days: int = 7) -> dict:
    db = get_database()
    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=days)
    pipeline = [
        {"$match": {"completed_at": {"$gte": cutoff}}},
        {"$group": {"_id": "$triage_status", "count": {"$sum": 1}}}
    ]
    triage_counts = {"green": 0, "yellow": 0, "red": 0}
    for row in db["checkin_history"].aggregate(pipeline):
        status = (row.get("_id") or "").lower()
        if status in triage_counts:
            triage_counts[status] = row.get("count", 0)

    senior_count = db["users"].count_documents({"role": "senior"})
    total_checkins = sum(triage_counts.values())
    alerts = triage_counts["yellow"] + triage_counts["red"]

    return {
        "total_seniors": senior_count,
        "total_checkins": total_checkins,
        "green": triage_counts["green"],
        "yellow": triage_counts["yellow"],
        "red": triage_counts["red"],
        "alerts": alerts,
        "window_days": days
    }
