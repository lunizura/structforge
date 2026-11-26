from __future__ import annotations


class StructForgeError(Exception):
    """Base exception for all structforge errors."""


class BinaryFormatError(StructForgeError):
    """Raised when input bytes are not a valid x86-64 PE32+ or ELF64 image."""


class SnapshotFormatError(StructForgeError):
    """Raised when a .sfdump binary snapshot is malformed or truncated."""


class InferenceError(StructForgeError):
    """Raised when struct layout or pointer-chain inference fails."""


class SpecValidationError(StructForgeError):
    """Raised when an SdkSpec JSON or identifier is invalid."""
