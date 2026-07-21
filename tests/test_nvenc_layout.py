"""NVENC ctypes struct layouts must byte-match the SDK 13.0 headers.

Ground-truth sizes and offsets come from a probe program compiled against
nvEncodeAPI.h (SDK 13.0). A mismatch means the driver reads or writes
outside the Python-allocated struct, so these are hard regression gates.
"""

import ctypes

from framepump.nvenc import bindings as b


def test_struct_sizes():
    assert ctypes.sizeof(b.NV_ENC_PIC_PARAMS) == 3360
    assert ctypes.sizeof(b.NV_ENC_CODEC_PIC_PARAMS) == 1544
    assert ctypes.alignment(b.NV_ENC_CODEC_PIC_PARAMS) == 8
    assert ctypes.sizeof(b.NV_ENC_LOCK_BITSTREAM) == 1544
    assert ctypes.sizeof(b.NV_ENC_INITIALIZE_PARAMS) == 1800
    assert ctypes.sizeof(b.NV_ENC_REGISTER_RESOURCE) == 1536


def test_pic_params_field_offsets():
    offsets = {
        'inputTimeStamp': 24,
        'inputBuffer': 40,
        'outputBitstream': 48,
        'bufferFmt': 64,
        'pictureType': 72,
        'codecPicParams': 80,
        'meHintCountsPerBlock': 1624,
        'meExternalHints': 1656,
        'qpDeltaMap': 1712,
        'meHintRefPicDist': 1728,
        'alphaBuffer': 1736,
    }
    for name, offset in offsets.items():
        assert getattr(b.NV_ENC_PIC_PARAMS, name).offset == offset, name
