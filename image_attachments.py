"""
Shared photo holder for multimodal agent turns.

The ReAct agent can only pass text in tool arguments, so Streamlit stores
uploaded/camera images here before invoke(); review_student_photo reads them.
Cleared after each turn so old photos never leak into later answers.
"""

from __future__ import annotations

_PENDING_IMAGES: list[dict] = []


def set_pending_images(images: list[dict] | None) -> None:
    """Store images for the current agent turn.

    Each item: ``{"bytes": b"...", "mime": "image/png"}``.
    """
    global _PENDING_IMAGES
    _PENDING_IMAGES = list(images or [])


def get_pending_images() -> list[dict]:
    """Return images attached for the current agent turn (may be empty)."""
    return _PENDING_IMAGES


def clear_pending_images() -> None:
    """Forget any attached images (call after each agent turn)."""
    global _PENDING_IMAGES
    _PENDING_IMAGES = []
