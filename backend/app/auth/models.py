"""Authenticated user context extracted from Supabase JWT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PlanTier = Literal["free", "pro"]


@dataclass(frozen=True)
class UserContext:
    user_id: str
    email: str
    plan: PlanTier = "free"
