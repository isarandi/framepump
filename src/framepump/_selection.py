"""Frame selection: which frames of the source video a view refers to.

A selection is either *symbolic* — a chain of not-yet-applied slices, usable
before the total frame count is known — or *resolved* — a concrete ``range``
over source frame indices. Symbolic selections let ``VideoFrames`` defer the
full packet scan of the video: a chain that reduces to a plain forward slice
can be decoded by streaming from the start without ever knowing the frame
count, while anything length-dependent (``len()``, negative indices or steps)
resolves the chain against the count once the index is built.

Slice composition is delegated to :mod:`slicecompose`, whose reductions are
sound for every sequence length; resolution applies the (reduced) chain to a
``range`` object, which is exact once the length is known.
"""

from __future__ import annotations

from slicecompose import SliceChain


class FrameSelection:
    """Immutable selection of frames, symbolic or resolved."""

    __slots__ = ('_chain', '_range')

    def __init__(self, chain: SliceChain | None = None, rng: range | None = None) -> None:
        if (chain is None) == (rng is None):
            raise ValueError('Exactly one of chain and rng must be given')
        self._chain = chain
        self._range = rng

    @classmethod
    def identity(cls) -> FrameSelection:
        """The whole video, frame count not yet known."""
        return cls(chain=SliceChain())

    @classmethod
    def from_range(cls, rng: range) -> FrameSelection:
        return cls(rng=rng)

    @property
    def is_resolved(self) -> bool:
        return self._range is not None

    @property
    def range(self) -> range:
        """The concrete source-index range (resolved selections only)."""
        if self._range is None:
            raise RuntimeError('Selection is symbolic; call resolve() first')
        return self._range

    def sliced(self, item: slice) -> FrameSelection:
        """Return the selection restricted by one more slice."""
        if self._range is not None:
            return FrameSelection(rng=self._range[item])
        return FrameSelection(chain=SliceChain(*self._chain.slices, item))

    def resolve(self, n_frames: int) -> FrameSelection:
        """Return the resolved equivalent given the total source frame count."""
        if self._range is not None:
            return self
        rng = range(n_frames)
        for s in self._chain.slices:
            rng = rng[s]
        return FrameSelection(rng=rng)

    @property
    def step_product(self) -> int:
        """Product of all slice steps: the effective stride (sign included).

        Well-defined without knowing the frame count, even for chains that
        slicecompose cannot reduce to a single slice.
        """
        if self._range is not None:
            return self._range.step
        product = 1
        for s in self._chain.slices:
            product *= s.step if s.step is not None else 1
        return product

    @property
    def streamable_slice(self) -> slice | None:
        """A single forward slice equivalent for every length, or None.

        When not None, the selection can be produced by decoding from the
        start and skipping (``islice`` semantics) without knowing the frame
        count. Resolved selections return None: they have an index behind
        them and use the seek-based paths instead.
        """
        if self._range is not None:
            return None
        slices = self._chain.slices
        if not slices:
            return slice(None)
        if len(slices) != 1:
            return None
        s = slices[0]
        if s.step is not None and s.step < 0:
            return None
        if s.start is not None and s.start < 0:
            return None
        if s.stop is not None and s.stop < 0:
            return None
        return s

    def __repr__(self) -> str:
        if self._range is not None:
            return f'FrameSelection({self._range!r})'
        return f'FrameSelection(chain={self._chain.slices!r})'
