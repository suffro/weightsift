"""Symbols of stored weight representations, and their exact inverses.

Every function here is a bijection on the stored bytes (tested over every pattern): fields are split for modelling and
merged back bit for bit. Nothing is converted to a floating-point value except where a context needs a magnitude, and
then only to build a context, never to reconstruct.

  BF16      uint16 patterns -> sign (1 bit), exponent (8 bits), mantissa (7 bits)
  int4      packed bytes -> nibbles, low nibble first (Colibri's and compressed-tensors' order); `signed_int4` reads a
            nibble as a quantization level under the container's declared encoding (offset-8 or two's complement)
  MXFP4     packed E2M1 nibbles (low first) + E8M0 block scales; magnitude index = nibble & 7, sign = nibble >> 3
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
# BF16


def bf16_fields(patterns: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(sign, exponent, mantissa) as uint8 arrays of the patterns' shape."""
    p = np.asarray(patterns, dtype=np.uint16)
    return (p >> 15).astype(np.uint8), ((p >> 7) & 0xFF).astype(np.uint8), (p & 0x7F).astype(np.uint8)


def bf16_merge(sign: np.ndarray, exponent: np.ndarray, mantissa: np.ndarray) -> np.ndarray:
    return ((sign.astype(np.uint16) << 15) | (exponent.astype(np.uint16) << 7) | mantissa.astype(np.uint16)).astype(np.uint16)


def bf16_high_low(patterns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The two bytes of each pattern: high (sign and the exponent's top 7 bits), low (the exponent's last bit and the
    mantissa). ZipNN's byte grouping."""
    p = np.asarray(patterns, dtype=np.uint16)
    return (p >> 8).astype(np.uint8), (p & 0xFF).astype(np.uint8)


# ---------------------------------------------------------------------------------------------------------------------
# int4


def unpack_nibbles(packed: np.ndarray) -> np.ndarray:
    """Packed bytes [..., n] -> nibbles [..., 2n], low nibble first (nib[2k] = b & 15, nib[2k+1] = b >> 4)."""
    b = np.asarray(packed, dtype=np.uint8)
    out = np.empty(b.shape[:-1] + (b.shape[-1] * 2,), dtype=np.uint8)
    out[..., 0::2] = b & 0x0F
    out[..., 1::2] = b >> 4
    return out


def pack_nibbles(nibbles: np.ndarray) -> np.ndarray:
    n = np.asarray(nibbles, dtype=np.uint8)
    if n.shape[-1] % 2:
        raise ValueError("an even number of nibbles per row is required")
    if int(n.max(initial=0)) > 15:
        raise ValueError("nibbles must be below 16")
    return (n[..., 0::2] | (n[..., 1::2] << 4)).astype(np.uint8)


INT4_ENCODINGS = ("offset8", "twos")


def signed_int4(nibbles: np.ndarray, encoding: str) -> np.ndarray:
    """Quantization levels in [-8, 7] (int8) of nibbles under a container's declared encoding."""
    n = np.asarray(nibbles, dtype=np.int16)
    if encoding == "offset8":
        return (n - 8).astype(np.int8)
    if encoding == "twos":
        return np.where(n >= 8, n - 16, n).astype(np.int8)
    raise ValueError(f"unknown int4 encoding {encoding!r}")


def detect_int4_encoding(nibbles: np.ndarray) -> str:
    """The encoding whose zero level is the most frequent nibble (both put a trained matrix's mode at level 0)."""
    hist = np.bincount(np.asarray(nibbles, dtype=np.uint8).ravel(), minlength=16)
    return "offset8" if hist[8] >= hist[0] else "twos"


# ---------------------------------------------------------------------------------------------------------------------
# MXFP4 (OCP microscaling: E2M1 elements, one E8M0 scale per block of 32)

E2M1_MAGNITUDES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def e2m1_fields(nibbles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sign, magnitude index 0..7) of E2M1 nibbles."""
    n = np.asarray(nibbles, dtype=np.uint8)
    return (n >> 3).astype(np.uint8), (n & 7).astype(np.uint8)


def e2m1_merge(sign: np.ndarray, magnitude: np.ndarray) -> np.ndarray:
    return ((sign.astype(np.uint8) << 3) | magnitude.astype(np.uint8)).astype(np.uint8)
