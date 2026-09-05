"""Canonical Huffman coding, written from scratch.

Stage 3 of the Deep Compression pipeline. No compression library is used: the
tree construction, the canonical code assignment, the bit packing and the
decoder are all implemented here.

Why Huffman helps at all
------------------------
After quantization the weight codes are far from uniformly distributed - weights
concentrate near zero, so the middle codes occur an order of magnitude more
often than the extremes. A fixed-width code spends b bits on every symbol
regardless of frequency. Huffman assigns short codes to frequent symbols, and
the achievable mean length is bounded below by the entropy H of the code
distribution:

        H(p)  <=  mean Huffman length  <  H(p) + 1     bits/symbol

so the saving over fixed-width coding is roughly (b - H) bits per symbol. For
4-bit weight codes on a trained network H is typically ~3.2-3.6 bits, i.e. a
10-20% saving on the value stream, on top of everything quantization already did.

Canonical Huffman
-----------------
An ordinary Huffman tree would have to be serialised to be decodable. Canonical
Huffman instead derives the codes deterministically from the *code lengths*
alone, so the stored table is one small integer per symbol rather than a tree.
That table is counted as storage overhead in `sizing.py`, as Q2(c) requires.
"""
from __future__ import annotations

import heapq
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Bits used to store one symbol's code length in the serialised table.
# 5 bits allows lengths up to 31, which is far above anything reachable for the
# small alphabets (<= 256 symbols) we encode.
CODE_LENGTH_FIELD_BITS = 5


@dataclass
class HuffmanCode:
    """A built canonical Huffman code for one symbol stream."""

    lengths: Dict[int, int]              # symbol -> code length in bits
    codes: Dict[int, str]                # symbol -> canonical code string
    freqs: Dict[int, int]                # symbol -> occurrence count
    alphabet_size: int                   # symbols the table must cover

    @property
    def table_bits(self) -> int:
        """Storage for the code table itself.

        Canonical Huffman is fully determined by the per-symbol code lengths, so
        the table is `alphabet_size` fixed-width length fields. Symbols that
        never occur get length 0 and still occupy a field, because the decoder
        must know the alphabet positions.
        """
        return self.alphabet_size * CODE_LENGTH_FIELD_BITS

    def payload_bits(self) -> int:
        """Bits in the encoded symbol stream (excluding the table)."""
        return sum(self.freqs[s] * self.lengths[s] for s in self.freqs)

    def total_bits(self) -> int:
        return self.payload_bits() + self.table_bits

    def mean_code_length(self) -> float:
        n = sum(self.freqs.values())
        return self.payload_bits() / n if n else 0.0


def _huffman_lengths(freqs: Dict[int, int]) -> Dict[int, int]:
    """Code length per symbol, from the standard two-queue Huffman construction.

    Repeatedly merge the two lowest-frequency nodes; each merge adds one bit to
    the depth of every symbol beneath it. The tie-breaking counter keeps the
    result deterministic, which matters because we want byte-identical output
    across runs for reproducibility.

    Degenerate case: a stream containing a single distinct symbol has zero
    entropy, but a code of length 0 cannot be written down, so that symbol is
    assigned length 1.
    """
    if not freqs:
        return {}
    if len(freqs) == 1:
        return {next(iter(freqs)): 1}

    counter = 0
    heap: List[Tuple[int, int, object]] = []
    for sym, f in sorted(freqs.items()):
        heapq.heappush(heap, (f, counter, [sym]))
        counter += 1

    depth: Dict[int, int] = {s: 0 for s in freqs}
    while len(heap) > 1:
        f1, _, syms1 = heapq.heappop(heap)
        f2, _, syms2 = heapq.heappop(heap)
        merged = syms1 + syms2
        for s in merged:
            depth[s] += 1                 # every symbol below the merge gets one more bit
        heapq.heappush(heap, (f1 + f2, counter, merged))
        counter += 1

    return depth


def _canonical_codes(lengths: Dict[int, int]) -> Dict[int, str]:
    """Assign canonical codes from code lengths.

    Rule: sort symbols by (length, symbol); walk them in order assigning
    consecutive binary values, shifting left whenever the length increases.
    The resulting code is prefix-free and reconstructible from lengths alone,
    which is what lets us store only the lengths.
    """
    used = {s: L for s, L in lengths.items() if L > 0}
    if not used:
        return {}
    codes: Dict[int, str] = {}
    code = 0
    prev_len = None
    for sym in sorted(used, key=lambda s: (used[s], s)):
        L = used[sym]
        if prev_len is None:
            prev_len = L
        else:
            code = (code + 1) << (L - prev_len)
            prev_len = L
        codes[sym] = format(code, f"0{L}b")
    return codes


def build_huffman(symbols: Sequence[int], alphabet_size: Optional[int] = None) -> HuffmanCode:
    """Build a canonical Huffman code for a symbol stream."""
    freqs = dict(Counter(int(s) for s in symbols))
    lengths = _huffman_lengths(freqs)
    codes = _canonical_codes(lengths)
    if alphabet_size is None:
        alphabet_size = (max(freqs) + 1) if freqs else 0
    return HuffmanCode(lengths=lengths, codes=codes, freqs=freqs,
                       alphabet_size=alphabet_size)


def encode(symbols: Sequence[int], code: HuffmanCode) -> str:
    """Encode a symbol stream to a bit string."""
    table = code.codes
    return "".join(table[int(s)] for s in symbols)


def decode(bitstring: str, code: HuffmanCode, num_symbols: int) -> List[int]:
    """Decode a bit string back into symbols.

    Walks the bit stream against a prefix-code lookup. Its only purpose is to
    prove the encoding is genuinely decodable - every size figure reported in
    the assignment is validated by a round trip through this function.
    """
    inverse = {c: s for s, c in code.codes.items()}
    out: List[int] = []
    buffer = ""
    for bit in bitstring:
        buffer += bit
        if buffer in inverse:
            out.append(inverse[buffer])
            buffer = ""
            if len(out) == num_symbols:
                break
    return out


def entropy_bits(symbols: Sequence[int]) -> float:
    """Shannon entropy of the symbol stream, in bits/symbol.

    This is the information-theoretic floor for any symbol-wise code. Reporting
    the achieved Huffman mean length against it shows how much of the available
    redundancy the coder actually captured (Huffman is within 1 bit of it, and
    usually far closer).
    """
    import math
    counts = Counter(int(s) for s in symbols)
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


@dataclass
class HuffmanResult:
    """Outcome of Huffman-coding one stream, with the numbers needed for sizing."""

    original_bits: int          # fixed-width cost of the same stream
    payload_bits: int           # Huffman-coded stream
    table_bits: int             # canonical code-length table
    total_bits: int             # payload + table
    mean_length: float          # achieved bits/symbol
    entropy: float              # theoretical floor, bits/symbol
    num_symbols: int
    verified: bool = False      # True once a decode round trip has succeeded
    code: Optional[HuffmanCode] = field(default=None, repr=False)


def huffman_compress(symbols: Sequence[int], fixed_width_bits: int,
                     alphabet_size: Optional[int] = None,
                     verify: bool = True) -> HuffmanResult:
    """Huffman-code a stream and report its exact cost.

    Args:
        symbols: the stream (quantized weight codes, or sparse index deltas).
        fixed_width_bits: what the same stream would cost without entropy
            coding, per symbol - the baseline the saving is measured against.
        alphabet_size: number of symbol slots the table must describe. Pass the
            full alphabet (e.g. 2^b) rather than the number of *observed*
            symbols, so the table cost is not understated.
        verify: run a decode round trip and assert exact recovery.

    Returns:
        HuffmanResult. `total_bits` is what `sizing.py` charges for the stream,
        and it always includes the table.
    """
    symbols = [int(s) for s in symbols]
    n = len(symbols)
    if n == 0:
        return HuffmanResult(0, 0, 0, 0, 0.0, 0.0, 0, verified=True)

    code = build_huffman(symbols, alphabet_size=alphabet_size)
    payload = code.payload_bits()
    table = code.table_bits

    verified = False
    if verify:
        bits = encode(symbols, code)
        assert len(bits) == payload, f"payload mismatch: {len(bits)} vs {payload}"
        verified = decode(bits, code, n) == symbols

    return HuffmanResult(
        original_bits=n * fixed_width_bits,
        payload_bits=payload,
        table_bits=table,
        total_bits=payload + table,
        mean_length=code.mean_code_length(),
        entropy=entropy_bits(symbols),
        num_symbols=n,
        verified=verified,
        code=code,
    )


def huffman_result_from_counts(counts: Dict[int, int], fixed_width_bits: int,
                               alphabet_size: int) -> HuffmanResult:
    """Cost a stream from its symbol *frequencies* alone, without materialising it.

    Huffman code lengths depend only on the frequency distribution, so a stream
    of 300,000 weight codes can be costed from a 16-entry histogram. Building the
    Python symbol list instead would dominate the runtime of the whole pipeline -
    the encoder searches seven layouts per layer across 53 layers, and
    materialising every candidate stream is what made the first implementation
    unusably slow.

    `build_huffman` remains the reference path and is exercised by the
    round-trip tests; this function must agree with it exactly, which
    scripts/test_huffman.py asserts.
    """
    counts = {int(s): int(c) for s, c in counts.items() if c > 0}
    n = sum(counts.values())
    if n == 0:
        return HuffmanResult(0, 0, 0, 0, 0.0, 0.0, 0, verified=True)

    lengths = _huffman_lengths(counts)
    payload = sum(counts[s] * lengths[s] for s in counts)
    table = alphabet_size * CODE_LENGTH_FIELD_BITS

    import math
    entropy = -sum((c / n) * math.log2(c / n) for c in counts.values())

    return HuffmanResult(
        original_bits=n * fixed_width_bits,
        payload_bits=payload,
        table_bits=table,
        total_bits=payload + table,
        mean_length=payload / n,
        entropy=entropy,
        num_symbols=n,
        verified=False,
        code=HuffmanCode(lengths=lengths, codes=_canonical_codes(lengths),
                         freqs=counts, alphabet_size=alphabet_size),
    )
