Safe Seek Points
================

Random access (``frames[i]``) seeks into the container and decodes
forward until the target frame comes out. For every frame, the index
stores one number — the *safe seek point* — and that number is the
timestamp that gets requested from the seek. This page explains what the
number is, and why the request is chosen so carefully instead of just
asking for the target's own timestamp.

The problem
-----------

Packets are stored in **decode order**. A B-frame is predicted from
frames that were decoded before it, so its packet comes *after* all its
reference frames in the file — including references that will be
*displayed* after it. Display timestamps (PTS) are therefore shuffled
relative to file order.

To produce one target frame, the decoder must start at a keyframe and
decode forward, packet by packet, until the target comes out. Which of
the intermediate frames the target truly references is codec-internal
knowledge; FramePump does not parse bitstreams to find out, it simply
decodes everything from the keyframe onward, which is safe for every
codec.

So the question is which keyframe to start from — and the container API
constrains how we can even express that. **You cannot seek to a file
position.** The only operation is a timestamp seek:
``seek(t, backward=True)`` means "position the stream at the last
keyframe whose timestamp is at most ``t``" (without ``backward=True``
the demuxer may land *after* ``t``, which is useless when you can only
decode forward). So "where should decoding start?" becomes "which
timestamp should we request?", and that request is what the index must
precompute.

A worked example
----------------

An ordinary B-frame stream; say the target is the frame with **PTS 4**:

::

    display order:   I0  B1  B2  P3  B4  B5  P6

    file position:     0      1      2      3      4      5      6
    packet (PTS):    I(0)   P(3)   B(1)   B(2)   P(6)   B(4)   B(5)
    running max:       0      3      3      3      6      6      6

                       ▲                    ▲             ▲
                       L                    S             j
                 landing keyframe     scan boundary   target packet

Three points play a role:

- **j — the target's packet** (position 5). The frame we want. Note that
  its reference P(6) is stored *before* it (position 4) even though it
  displays after it — B-frame reordering at work.

- **L — where decoding must start** (position 0). Reading forward from
  this keyframe passes through everything the target needs: I(0), P(3),
  P(6) and B(4) itself. Making the seek land here, or earlier, is the
  whole goal.

- **S — the scan boundary** (position 3): the last position in the file
  before anything that displays *later* than the target has appeared.
  The index stores the PTS found at S — here **2** — and that is the
  value later handed to the seek.

The access then plays out like this: FramePump requests
``seek(2, backward=True)``, the demuxer lands on the last keyframe with
timestamp ≤ 2 — that is I(0), which is L — and decoding reads forward,
straight past S and through j, until the frame with PTS 4 is emitted.
S itself is never a place anyone decodes from or to; its PTS exists only
to shape the request. The rest of this page explains why this particular
request is the right one.

What a keyframe does and does not guarantee
-------------------------------------------

A natural mental model is that a keyframe is an absolute barrier:
nothing stored after it should ever need anything stored before it.
That model is exactly right for **closed-GOP (IDR)** keyframes — the
decoder's reference buffer is flushed, so referencing across is
impossible.

But codecs also allow **open-GOP** keyframes, which permit one precisely
delimited exception: **leading frames** — frames stored *after* the
keyframe but *displayed before* it — may still reference the previous
group of pictures. Encoders use this because it compresses better.
Frames that display at or after the keyframe may never reference across
it; that restriction is what makes the keyframe a valid random access
point at all.

::

    storage order:   ...  P(27)   K(30)   B(28)   B(29)   P(33)  ...
    display order:   ...   27      28      29      30      33    ...

B(28) and B(29) sit after K(30) in the file, display before it, and may
be predicted from P(27) in the previous GOP. Starting decode at K(30)
cannot produce them correctly. P(33) and everything displaying after 30,
on the other hand, is guaranteed decodable from K(30).

So the boundary rule is decided by comparing the *target's* display time
with the *keyframe's* display time — the timestamps of the older packets
play no role:

- Target displays **at or after** the keyframe → nothing stored before
  the keyframe is needed, period.
- Target displays **before** the keyframe (a leading frame) → the
  keyframe is not a valid starting point; decoding must start one
  keyframe earlier.

Why the stored request lands correctly
--------------------------------------

Check the two cases of the boundary rule against what the seek does with
the stored request:

**Ordinary targets.** The demuxer lands on a keyframe with timestamp at
most the request, which is at most the target's PTS — so the target
displays at or after the landing keyframe. By the boundary rule, nothing
stored before that keyframe is needed, and decoding forward sweeps up
everything that is.

**Leading targets.** Here the scan does something automatically that is
easy to miss: the enclosing keyframe *displays later than the target* —
that is the definition of "leading" — so the moment K(30) appears in the
file, the running max jumps past the target's PTS and the streak ends.
The boundary S for B(28) therefore falls *before* K(30) in the file, at
P(27), and the stored request is 27. At seek time, K(30) fails the
"timestamp ≤ 27" test, and the demuxer falls through to the *previous*
keyframe — exactly where the backward references of a leading frame are
reachable. The dangerous case defuses itself, and the scan never had to
look at a keyframe flag or a reference list to make it happen.

Why not simply request the target's own PTS?
--------------------------------------------

Two simpler strategies are worth ruling out explicitly.

*Seeking by storage order* — "start at the last keyframe stored before
the target's packet" — is genuinely wrong: for a leading frame that
picks the enclosing keyframe, which cannot decode it. This is the trap
of reasoning in file positions.

*Requesting the target's PTS* — ``seek(T, backward=True)`` — is much
better, and for well-behaved files it lands on the same keyframe as the
stored request. The catch is that "seek by timestamp" is not one
well-defined operation. Each demuxer resolves the request against its
own tables, in its own timeline: MP4 sample tables are DTS-based, MKV
cues have cluster granularity, MPEG-TS interpolates byte positions. DTS
and PTS differ exactly when B-frames reorder — the very streams where
precision matters — so a request at the target's PTS can be interpreted
against a timeline where it selects a keyframe you meant to exclude.

The two failure directions are not symmetric:

- **Landing too early** costs a few extra decoded frames. Always
  correct, merely slower.
- **Landing too late** silently loses reference frames. The decoded
  target still carries the right PTS, so no downstream check can tell
  the pixels are wrong — or the target's packet is never reached at all.

The safe seek point is the request with one-sided risk: it is the last
timestamp the scan observed *before the stream had delivered anything
past the target*. Whatever timeline a demuxer compares it against, a
value from that early in the stream can only move the landing earlier,
never later. The margin converts an unknowable per-container risk into
a small, bounded decoding cost.

Implementation notes
--------------------

The index scan exists independently of seeking: containers have no
"frame number" API, so serving ``frames[42]`` requires translating 42
into a PTS, which takes the full display-order PTS list — the same scan
also yields the exact ``len()``, VFR timing, the CFR source map and
reverse-iteration planning. The safe seek point is one extra column
derived from data the scan already collects: the running max never
decreases, so each frame's boundary is found by a binary search over the
recorded values, once per frame, at build time.

The landing point only determines *which packets get decoded*; frame
exactness comes from the final step — decoding forward and comparing
each emitted frame's PTS against the target PTS from the index.

All of this trusts the container's timestamps and keyframe flags to be
truthful. Files where they are not are caught by content verification at
open time, which degrades them to always-correct decode-from-start
access (see :doc:`frame-indexing`). Non-seekable streams skip seeking
entirely: every safe seek point is zero and access decodes from the
start.
