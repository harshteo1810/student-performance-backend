import os

from fastapi import Header, HTTPException, status

API_KEY = os.getenv("API_KEY")  # set this in Render's Environment tab


def verify_api_key(x_api_key: str = Header(..., description="Shared API key")):
    """
    Lightweight protection, not real auth: a single shared secret checked
    against the X-API-Key request header. Fine for a college project to
    keep random internet traffic off your free-tier instance -- NOT a
    substitute for per-user authentication if this ever handles real
    student data with multiple real users.

    If API_KEY isn't set as an env var (e.g. local dev without bothering
    to configure it), this dependency is skipped entirely so local
    testing isn't blocked.
    """
    if API_KEY is None:
        return  # no key configured -- open access (local dev convenience)
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
