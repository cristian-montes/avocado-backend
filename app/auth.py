"""Authentication via Clerk JWT verification.

Access is invitation-only: Clerk's "Restricted" mode ensures only invited users can
authenticate and receive a token. Therefore any valid Clerk JWT is an authorized user
and no separate allowlist is required here.
"""
import logging

import jwt
from fastapi import Header, HTTPException

from app import config

logger = logging.getLogger(__name__)

jwks_client = jwt.PyJWKClient(config.JWKS_URL) if config.JWKS_URL else None


def verify_clerk_user(authorization: str = Header(None)) -> str:
    """Validate the Clerk JWT and return the authenticated user id.

    Raises HTTPException(401) on missing/invalid tokens and HTTPException(503)
    when the auth backend is not configured.
    """
    auth_parts = authorization.split() if authorization else []
    if len(auth_parts) != 2 or auth_parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing or invalid security token")

    if not jwks_client or not config.JWT_ISSUER or not config.JWT_AUDIENCE:
        logger.error("JWT configuration is incomplete")
        raise HTTPException(status_code=503, detail="Authentication is unavailable")

    token = auth_parts[1]

    try:
        signing_key = jwks_client.get_signing_key_from_jwt(token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=config.JWT_ISSUER,
            audience=config.JWT_AUDIENCE,
            leeway=30,
        )
        return payload.get("sub")
    except HTTPException:
        raise
    except Exception as e:
        logger.info("JWT verification failed: %s", type(e).__name__)
        raise HTTPException(status_code=401, detail="Invalid security token") from e
