"""Shared dependencies and utilities."""

from app.db import get_database


def get_checkins_collection():
    """Get the checkin_history MongoDB collection."""
    return get_database()["checkin_history"]


def get_screenings_collection():
    """Get the screenings MongoDB collection."""
    return get_database()["screenings"]


def get_users_collection():
    """Get the users MongoDB collection."""
    return get_database()["users"]
