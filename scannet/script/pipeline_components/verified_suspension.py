"""Require real suspension evidence before expanding an upper object surface."""
from . import suspension_geometry


def construct(context):
    suspension_geometry.construct(context, require_suspension=True)
