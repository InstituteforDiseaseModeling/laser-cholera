"""Stub `compute` re-export — falls back to a pure-Python implementation when no C extension is available.

The optional `_core` C extension exposes a fast `compute(args)`. When
the extension is not present (e.g. plain wheel install without the
C build), a trivial pure-Python fallback that returns the longest
argument is provided so importers do not have to branch on the
extension's availability.
"""

try:
    from ._core import compute
except ImportError:

    def compute(args: list) -> object:
        """Return the longest element of `args` (pure-Python fallback for `_core.compute`).

        Args:
            args: An iterable of items supporting `len()`.

        Returns:
            The element of `args` with the maximum `len()`.
        """
        return max(args, key=len)
