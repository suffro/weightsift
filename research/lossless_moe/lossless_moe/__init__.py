"""Lossless MoE weight compression: a feasibility study (research code, not a runtime).

Measures how much exploitable statistical structure routed-expert tensors hold beyond a static order-0 entropy coder,
on exactly the bytes a checkpoint stores, with every model and table cost counted. Nothing in `src/` imports it.
"""
