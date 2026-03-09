"""Low-level ctypes bindings for NVIDIA NVENC API."""

from __future__ import annotations

import ctypes
from ctypes import (
    CFUNCTYPE, POINTER, Structure, Union,
    byref, c_int8, c_int32, c_uint8, c_uint16, c_uint32, c_uint64, c_void_p,
)

from framepump.nvenc.exceptions import NvencError, NvencNotAvailable

# =============================================================================
# Constants
# =============================================================================

NVENCAPI_MAJOR_VERSION = 13
NVENCAPI_MINOR_VERSION = 0
NVENCAPI_VERSION = NVENCAPI_MAJOR_VERSION | (NVENCAPI_MINOR_VERSION << 24)


def _struct_version(ver: int) -> int:
    return NVENCAPI_VERSION | (ver << 16) | (0x7 << 28)


# Device types
NV_ENC_DEVICE_TYPE_CUDA = 1
NV_ENC_DEVICE_TYPE_OPENGL = 2

# Resource types
NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR = 1
NV_ENC_INPUT_RESOURCE_TYPE_CUDAARRAY = 2
NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX = 3

# Buffer formats
NV_ENC_BUFFER_FORMAT_NV12 = 0x00000001
NV_ENC_BUFFER_FORMAT_IYUV = 0x00000100  # I420: Y + U + V planar (4:2:0)
NV_ENC_BUFFER_FORMAT_YUV444 = 0x00001000  # Y + U + V planar (4:4:4)
NV_ENC_BUFFER_FORMAT_NV16 = 0x40000001  # Semi-planar Y + interleaved UV (4:2:2)
NV_ENC_BUFFER_FORMAT_ABGR = 0x10000000

# Picture structure
NV_ENC_PIC_STRUCT_FRAME = 0x01

# Flags
NV_ENC_PIC_FLAG_EOS = 0x8

# Rate control modes
NV_ENC_PARAMS_RC_CONSTQP = 0x0  # Constant QP - direct quality control
NV_ENC_PARAMS_RC_VBR = 0x1
NV_ENC_PARAMS_RC_CBR = 0x2

# Status codes
NV_ENC_SUCCESS = 0
NV_ENC_ERR_NEED_MORE_INPUT = 17  # Encoder needs more input to produce output

# Picture types (from NV_ENC_LOCK_BITSTREAM.pictureType)
NV_ENC_PIC_TYPE_IDR = 0x03  # IDR frame (keyframe)

# Buffer usage
NV_ENC_INPUT_IMAGE = 0

# OpenGL constants
GL_TEXTURE_2D = 0x0DE1

# Structure versions
NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER = _struct_version(1)
NV_ENC_INITIALIZE_PARAMS_VER = _struct_version(7) | (1 << 31)
NV_ENC_CONFIG_VER = _struct_version(9) | (1 << 31)
NV_ENC_PRESET_CONFIG_VER = _struct_version(5) | (1 << 31)
NV_ENC_CREATE_BITSTREAM_BUFFER_VER = _struct_version(1)
NV_ENC_PIC_PARAMS_VER = _struct_version(7) | (1 << 31)
NV_ENC_LOCK_BITSTREAM_VER = _struct_version(2) | (1 << 31)
NV_ENC_REGISTER_RESOURCE_VER = _struct_version(5)
NV_ENC_MAP_INPUT_RESOURCE_VER = _struct_version(4)


# =============================================================================
# ctypes Structures
# =============================================================================

class GUID(Structure):
    _fields_ = [
        ('Data1', c_uint32),
        ('Data2', c_uint16),
        ('Data3', c_uint16),
        ('Data4', c_uint8 * 8),
    ]


def _make_guid(d1: int, d2: int, d3: int, d4: list[int]) -> GUID:
    g = GUID()
    g.Data1, g.Data2, g.Data3 = d1, d2, d3
    g.Data4 = (c_uint8 * 8)(*d4)
    return g


# Codec GUIDs
NV_ENC_CODEC_H264_GUID = _make_guid(
    0x6bc82762, 0x4e63, 0x4ca4,
    [0xaa, 0x85, 0x1e, 0x50, 0xf3, 0x21, 0xf6, 0xbf]
)

# H.264 High 4:2:2 profile (required for YUV422 encoding)
NV_ENC_H264_PROFILE_HIGH_422_GUID = _make_guid(
    0xff3242e9, 0x613c, 0x4295,
    [0xa1, 0xe8, 0x2a, 0x7f, 0xe9, 0x4d, 0x81, 0x33]
)

# H.264 High 4:4:4 Predictive profile (required for YUV444 encoding)
NV_ENC_H264_PROFILE_HIGH_444_GUID = _make_guid(
    0x7ac663cb, 0xa598, 0x4960,
    [0xb8, 0x44, 0x33, 0x9b, 0x26, 0x1a, 0x7d, 0x52]
)

# Preset GUIDs
NV_ENC_PRESET_P4_GUID = _make_guid(
    0x90a7b826, 0xdf06, 0x4862,
    [0xb9, 0xd2, 0xcd, 0x6d, 0x73, 0xa0, 0x86, 0x81]
)
NV_ENC_PRESET_P7_GUID = _make_guid(
    0x84848c12, 0x6f71, 0x4c13,
    [0x93, 0x1b, 0x53, 0xe2, 0x83, 0xf5, 0x79, 0x74]
)

# Tuning info
NV_ENC_TUNING_INFO_UNDEFINED = 0
NV_ENC_TUNING_INFO_HIGH_QUALITY = 1
NV_ENC_TUNING_INFO_LOW_LATENCY = 2
NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY = 3
NV_ENC_TUNING_INFO_LOSSLESS = 4


class NV_ENCODE_API_FUNCTION_LIST(Structure):
    _fields_ = [
        ('version', c_uint32), ('reserved', c_uint32),
        ('nvEncOpenEncodeSession', c_void_p), ('nvEncGetEncodeGUIDCount', c_void_p),
        ('nvEncGetEncodeProfileGUIDCount', c_void_p), ('nvEncGetEncodeProfileGUIDs', c_void_p),
        ('nvEncGetEncodeGUIDs', c_void_p), ('nvEncGetInputFormatCount', c_void_p),
        ('nvEncGetInputFormats', c_void_p), ('nvEncGetEncodeCaps', c_void_p),
        ('nvEncGetEncodePresetCount', c_void_p), ('nvEncGetEncodePresetGUIDs', c_void_p),
        ('nvEncGetEncodePresetConfig', c_void_p), ('nvEncInitializeEncoder', c_void_p),
        ('nvEncCreateInputBuffer', c_void_p), ('nvEncDestroyInputBuffer', c_void_p),
        ('nvEncCreateBitstreamBuffer', c_void_p), ('nvEncDestroyBitstreamBuffer', c_void_p),
        ('nvEncEncodePicture', c_void_p), ('nvEncLockBitstream', c_void_p),
        ('nvEncUnlockBitstream', c_void_p), ('nvEncLockInputBuffer', c_void_p),
        ('nvEncUnlockInputBuffer', c_void_p), ('nvEncGetEncodeStats', c_void_p),
        ('nvEncGetSequenceParams', c_void_p), ('nvEncRegisterAsyncEvent', c_void_p),
        ('nvEncUnregisterAsyncEvent', c_void_p), ('nvEncMapInputResource', c_void_p),
        ('nvEncUnmapInputResource', c_void_p), ('nvEncDestroyEncoder', c_void_p),
        ('nvEncInvalidateRefFrames', c_void_p), ('nvEncOpenEncodeSessionEx', c_void_p),
        ('nvEncRegisterResource', c_void_p), ('nvEncUnregisterResource', c_void_p),
        ('nvEncReconfigureEncoder', c_void_p), ('reserved1', c_void_p),
        ('nvEncCreateMVBuffer', c_void_p), ('nvEncDestroyMVBuffer', c_void_p),
        ('nvEncRunMotionEstimationOnly', c_void_p), ('nvEncGetLastErrorString', c_void_p),
        ('nvEncSetIOCudaStreams', c_void_p), ('nvEncGetEncodePresetConfigEx', c_void_p),
        ('nvEncGetSequenceParamEx', c_void_p), ('nvEncRestoreEncoderState', c_void_p),
        ('nvEncLookaheadPicture', c_void_p), ('reserved2', c_void_p * 275),
    ]


class NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS(Structure):
    _fields_ = [
        ('version', c_uint32), ('deviceType', c_uint32), ('device', c_void_p),
        ('reserved', c_void_p), ('apiVersion', c_uint32),
        ('reserved1', c_uint32 * 253), ('reserved2', c_void_p * 64),
    ]


class NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE(Structure):
    # Bitfields packed: numCandsPerBlk16x16:4, numCandsPerBlk16x8:4, numCandsPerBlk8x16:4,
    # numCandsPerBlk8x8:4, numCandsPerSb:8, reserved:8 = 32 bits
    # Plus reserved1[3] = 12 bytes
    # Total = 16 bytes
    _fields_ = [('bitfields', c_uint32), ('reserved1', c_uint32 * 3)]


class NV_ENC_QP(Structure):
    _fields_ = [('qpInterP', c_uint32), ('qpInterB', c_uint32), ('qpIntra', c_uint32)]


class NV_ENC_RC_PARAMS(Structure):
    # Note: enableMinQP through reservedBitFields are bitfields packed into single uint32 in C++
    _fields_ = [
        ('version', c_uint32), ('rateControlMode', c_uint32), ('constQP', NV_ENC_QP),
        ('averageBitRate', c_uint32), ('maxBitRate', c_uint32), ('vbvBufferSize', c_uint32),
        ('vbvInitialDelay', c_uint32),
        # Bitfields: enableMinQP:1, enableMaxQP:1, enableInitialRCQP:1, enableAQ:1,
        # reservedBitField1:1, enableLookahead:1, disableIadapt:1, disableBadapt:1,
        # enableTemporalAQ:1, zeroReorderDelay:1, enableNonRefP:1, strictGOPTarget:1,
        # aqStrength:4, enableExtLookahead:1, reservedBitFields:15 = 32 bits
        ('rcFlags', c_uint32),
        ('minQP', NV_ENC_QP), ('maxQP', NV_ENC_QP), ('initialRCQP', NV_ENC_QP),
        ('temporallayerIdxMask', c_uint32),
        ('temporalLayerQP', c_uint8 * 8),
        ('targetQuality', c_uint8), ('targetQualityLSB', c_uint8),
        ('lookaheadDepth', c_uint16),
        ('lowDelayKeyFrameScale', c_uint8),
        ('yDcQPIndexOffset', c_int8), ('uDcQPIndexOffset', c_int8), ('vDcQPIndexOffset', c_int8),
        ('qpMapMode', c_uint32), ('multiPass', c_uint32), ('alphaLayerBitrateRatio', c_uint32),
        ('cbQPIndexOffset', c_int8), ('crQPIndexOffset', c_int8), ('reserved2', c_uint16),
        ('lookaheadLevel', c_uint32),
        ('reserved', c_uint32 * 3),
    ]


class NV_ENC_CONFIG_H264_VUI_PARAMETERS(Structure):
    _fields_ = [
        ('overscanInfoPresentFlag', c_uint32), ('overscanInfo', c_uint32),
        ('videoSignalTypePresentFlag', c_uint32), ('videoFormat', c_uint32),
        ('videoFullRangeFlag', c_uint32), ('colourDescriptionPresentFlag', c_uint32),
        ('colourPrimaries', c_uint32), ('transferCharacteristics', c_uint32),
        ('colourMatrix', c_uint32), ('chromaSampleLocationFlag', c_uint32),
        ('chromaSampleLocationTop', c_uint32), ('chromaSampleLocationBot', c_uint32),
        ('bitstreamRestrictionFlag', c_uint32), ('reserved', c_uint32 * 15),
    ]


class NV_ENC_CONFIG_H264(Structure):
    # Note: First 22 flags + 10 reserved bits are packed into a single uint32 in C++
    _fields_ = [
        # Bitfields: enableTemporalSVC:1, enableStereoMVC:1, hierarchicalPFrames:1,
        # hierarchicalBFrames:1, outputBufferingPeriodSEI:1, outputPictureTimingSEI:1,
        # outputAUD:1, disableSPSPPS:1, outputFramePackingSEI:1, outputRecoveryPointSEI:1,
        # enableIntraRefresh:1, enableConstrainedEncoding:1, repeatSPSPPS:1, enableVFR:1,
        # enableLTR:1, qpPrimeYZeroTransformBypassFlag:1, useConstrainedIntraPred:1,
        # enableFillerDataInsertion:1, disableSVCPrefixNalu:1, enableScalabilityInfoSEI:1,
        # singleSliceIntraRefresh:1, enableTimeCode:1, reservedBitFields:10 = 32 bits
        ('h264Flags', c_uint32),
        ('level', c_uint32), ('idrPeriod', c_uint32), ('separateColourPlaneFlag', c_uint32),
        ('disableDeblockingFilterIDC', c_uint32), ('numTemporalLayers', c_uint32),
        ('spsId', c_uint32), ('ppsId', c_uint32), ('adaptiveTransformMode', c_uint32),
        ('fmoMode', c_uint32), ('bdirectMode', c_uint32), ('entropyCodingMode', c_uint32),
        ('stereoMode', c_uint32), ('intraRefreshPeriod', c_uint32), ('intraRefreshCnt', c_uint32),
        ('maxNumRefFrames', c_uint32), ('sliceMode', c_uint32), ('sliceModeData', c_uint32),
        ('h264VUIParameters', NV_ENC_CONFIG_H264_VUI_PARAMETERS), ('ltrNumFrames', c_uint32),
        ('ltrTrustMode', c_uint32), ('chromaFormatIDC', c_uint32), ('maxTemporalLayers', c_uint32),
        ('useBFramesAsRef', c_uint32), ('numRefL0', c_uint32), ('numRefL1', c_uint32),
        # Additional fields after numRefL1: outputBitDepth, inputBitDepth, tfLevel
        ('outputBitDepth', c_uint32), ('inputBitDepth', c_uint32), ('tfLevel', c_uint32),
        ('reserved1', c_uint32 * 264), ('reserved2', c_void_p * 64),
    ]


class NV_ENC_CODEC_CONFIG(Union):
    _fields_ = [('h264Config', NV_ENC_CONFIG_H264), ('reserved', c_uint32 * 320)]


class NV_ENC_CONFIG(Structure):
    _fields_ = [
        ('version', c_uint32), ('profileGUID', GUID), ('gopLength', c_uint32),
        ('frameIntervalP', c_int32), ('monoChromeEncoding', c_uint32),
        ('frameFieldMode', c_uint32), ('mvPrecision', c_uint32), ('rcParams', NV_ENC_RC_PARAMS),
        ('encodeCodecConfig', NV_ENC_CODEC_CONFIG), ('reserved', c_uint32 * 278),
        ('reserved2', c_void_p * 64),
    ]


class NV_ENC_PRESET_CONFIG(Structure):
    _fields_ = [
        ('version', c_uint32), ('presetCfg', NV_ENC_CONFIG),
        ('reserved1', c_uint32 * 255), ('reserved2', c_void_p * 64),
    ]


class NV_ENC_INITIALIZE_PARAMS(Structure):
    # Note: In C++, fields after enablePTD are bitfields packed into a single uint32.
    # We use a combined bitfield uint32 to match the C++ layout.
    _fields_ = [
        ('version', c_uint32), ('encodeGUID', GUID), ('presetGUID', GUID),
        ('encodeWidth', c_uint32), ('encodeHeight', c_uint32), ('darWidth', c_uint32),
        ('darHeight', c_uint32), ('frameRateNum', c_uint32), ('frameRateDen', c_uint32),
        ('enableEncodeAsync', c_uint32), ('enablePTD', c_uint32),
        # Bitfields: reportSliceOffsets:1, enableSubFrameWrite:1, enableExternalMEHints:1,
        # enableMEOnlyMode:1, enableWeightedPrediction:1, splitEncodeMode:4,
        # enableOutputInVidmem:1, enableReconFrameOutput:1, enableOutputStats:1,
        # enableUniDirectionalB:1, reservedBitFields:19 = 32 bits total
        ('encodeFlags', c_uint32),
        ('privDataSize', c_uint32), ('reserved', c_uint32), ('privData', c_void_p),
        ('encodeConfig', POINTER(NV_ENC_CONFIG)),
        ('maxEncodeWidth', c_uint32), ('maxEncodeHeight', c_uint32),
        ('maxMEHintCountsPerBlock', NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE * 2),
        ('tuningInfo', c_uint32),
        ('bufferFormat', c_uint32),  # Input buffer format (DX12 only)
        ('reserved2', c_uint32 * 285), ('reserved3', c_void_p * 64),
    ]


class NV_ENC_CREATE_BITSTREAM_BUFFER(Structure):
    _fields_ = [
        ('version', c_uint32), ('size', c_uint32), ('memoryHeap', c_uint32),
        ('reserved', c_uint32), ('bitstreamBuffer', c_void_p), ('bitstreamBufferPtr', c_void_p),
        ('reserved1', c_uint32 * 58), ('reserved2', c_void_p * 64),
    ]


class NV_ENC_INPUT_RESOURCE_OPENGL_TEX(Structure):
    _fields_ = [
        ('texture', c_uint32),
        ('target', c_uint32),
    ]


class NV_ENC_REGISTER_RESOURCE(Structure):
    _fields_ = [
        ('version', c_uint32), ('resourceType', c_uint32),
        ('width', c_uint32), ('height', c_uint32), ('pitch', c_uint32),
        ('subResourceIndex', c_uint32), ('resourceToRegister', c_void_p),
        ('registeredResource', c_void_p), ('bufferFormat', c_uint32), ('bufferUsage', c_uint32),
        ('pInputFencePoint', c_void_p),
        ('chromaOffset', c_uint32 * 2), ('chromaOffsetIn', c_uint32 * 2),
        ('reserved1', c_uint32 * 244), ('reserved2', c_void_p * 61),
    ]


class NV_ENC_MAP_INPUT_RESOURCE(Structure):
    _fields_ = [
        ('version', c_uint32), ('subResourceIndex', c_uint32),
        ('inputResource', c_void_p), ('registeredResource', c_void_p),
        ('mappedResource', c_void_p), ('mappedBufferFmt', c_uint32),
        ('reserved1', c_uint32 * 251), ('reserved2', c_void_p * 63),
    ]


class NV_ENC_CODEC_PIC_PARAMS(Union):
    _fields_ = [('reserved', c_uint32 * 256)]


class NV_ENC_PIC_PARAMS(Structure):
    _fields_ = [
        ('version', c_uint32), ('inputWidth', c_uint32), ('inputHeight', c_uint32),
        ('inputPitch', c_uint32), ('encodePicFlags', c_uint32), ('frameIdx', c_uint32),
        ('inputTimeStamp', c_uint64), ('inputDuration', c_uint64), ('inputBuffer', c_void_p),
        ('outputBitstream', c_void_p), ('completionEvent', c_void_p), ('bufferFmt', c_uint32),
        ('pictureStruct', c_uint32), ('pictureType', c_uint32),
        ('codecPicParams', NV_ENC_CODEC_PIC_PARAMS),
        ('meHintCountsPerBlock', NVENC_EXTERNAL_ME_HINT_COUNTS_PER_BLOCKTYPE * 2),
        ('meExternalHints', c_void_p),
        ('reserved1', c_uint32 * 6), ('reserved2', c_void_p * 2), ('qpDeltaMap', c_void_p),
        ('qpDeltaMapSize', c_uint32), ('reservedBitFields', c_uint32),
        ('meHintRefPicDist', c_uint16 * 2),
        ('alphaBuffer', c_void_p), ('reserved3', c_uint32 * 286), ('reserved4', c_void_p * 59),
    ]


class NV_ENC_LOCK_BITSTREAM(Structure):
    _fields_ = [
        ('version', c_uint32), ('doNotWait', c_uint32),
        ('outputBitstream', c_void_p), ('sliceOffsets', c_void_p),
        ('frameIdx', c_uint32), ('hwEncodeStatus', c_uint32),
        ('numSlices', c_uint32), ('bitstreamSizeInBytes', c_uint32),
        ('outputTimeStamp', c_uint64), ('outputDuration', c_uint64),
        ('bitstreamBufferPtr', c_void_p),
        ('pictureType', c_uint32), ('pictureStruct', c_uint32),
        ('frameAvgQP', c_uint32), ('frameSatd', c_uint32),
        ('ltrFrameIdx', c_uint32), ('ltrFrameBitmap', c_uint32), ('temporalId', c_uint32),
        ('intraMBCount', c_uint32), ('interMBCount', c_uint32),
        ('averageMVX', c_int32), ('averageMVY', c_int32),
        ('alphaLayerSizeInBytes', c_uint32),
        ('outputStatsPtrSize', c_uint32), ('reserved', c_uint32),
        ('outputStatsPtr', c_void_p), ('frameIdxDisplay', c_uint32),
        ('reserved1', c_uint32 * 219), ('reserved2', c_void_p * 63),
        ('reservedInternal', c_uint32 * 8),
    ]


# =============================================================================
# NVENC API Wrapper
# =============================================================================

class NvencAPI:
    """Low-level wrapper for NVENC library functions."""

    _instance: NvencAPI | None = None
    _initialized: bool
    _lib: ctypes.CDLL
    _api: NV_ENCODE_API_FUNCTION_LIST

    def __new__(cls) -> NvencAPI:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._load_library()
        self._get_api()

    def _load_library(self) -> None:
        try:
            self._lib = ctypes.CDLL('libnvidia-encode.so.1')
        except OSError:
            try:
                self._lib = ctypes.CDLL('libnvidia-encode.so')
            except OSError as e:
                raise NvencNotAvailable(
                    'Could not load libnvidia-encode.so. '
                    'Ensure NVIDIA drivers are installed.'
                ) from e

    def _get_api(self) -> None:
        self._api = NV_ENCODE_API_FUNCTION_LIST()
        self._api.version = _struct_version(2)

        create_instance = self._lib.NvEncodeAPICreateInstance
        create_instance.argtypes = [POINTER(NV_ENCODE_API_FUNCTION_LIST)]
        create_instance.restype = c_uint32

        status = create_instance(byref(self._api))
        if status != NV_ENC_SUCCESS:
            raise NvencError(f'Failed to create NVENC API instance: {status}')

        # Create function prototypes
        self.nvEncOpenEncodeSessionEx = CFUNCTYPE(
            c_uint32, POINTER(NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS), POINTER(c_void_p)
        )(self._api.nvEncOpenEncodeSessionEx)

        self.nvEncGetEncodeGUIDCount = CFUNCTYPE(
            c_uint32, c_void_p, POINTER(c_uint32)
        )(self._api.nvEncGetEncodeGUIDCount)

        self.nvEncGetEncodeGUIDs = CFUNCTYPE(
            c_uint32, c_void_p, POINTER(GUID), c_uint32, POINTER(c_uint32)
        )(self._api.nvEncGetEncodeGUIDs)

        self.nvEncGetEncodePresetCount = CFUNCTYPE(
            c_uint32, c_void_p, GUID, POINTER(c_uint32)
        )(self._api.nvEncGetEncodePresetCount)

        self.nvEncGetEncodePresetGUIDs = CFUNCTYPE(
            c_uint32, c_void_p, GUID, POINTER(GUID), c_uint32, POINTER(c_uint32)
        )(self._api.nvEncGetEncodePresetGUIDs)

        self.nvEncGetEncodePresetConfig = CFUNCTYPE(
            c_uint32, c_void_p, GUID, GUID, POINTER(NV_ENC_PRESET_CONFIG)
        )(self._api.nvEncGetEncodePresetConfig)

        # nvEncGetEncodePresetConfigEx includes tuningInfo parameter
        self.nvEncGetEncodePresetConfigEx = CFUNCTYPE(
            c_uint32, c_void_p, GUID, GUID, c_uint32, POINTER(NV_ENC_PRESET_CONFIG)
        )(self._api.nvEncGetEncodePresetConfigEx)

        self.nvEncGetInputFormatCount = CFUNCTYPE(
            c_uint32, c_void_p, GUID, POINTER(c_uint32)
        )(self._api.nvEncGetInputFormatCount)

        self.nvEncGetInputFormats = CFUNCTYPE(
            c_uint32, c_void_p, GUID, POINTER(c_uint32), c_uint32, POINTER(c_uint32)
        )(self._api.nvEncGetInputFormats)

        self.nvEncInitializeEncoder = CFUNCTYPE(
            c_uint32, c_void_p, POINTER(NV_ENC_INITIALIZE_PARAMS)
        )(self._api.nvEncInitializeEncoder)

        self.nvEncCreateBitstreamBuffer = CFUNCTYPE(
            c_uint32, c_void_p, POINTER(NV_ENC_CREATE_BITSTREAM_BUFFER)
        )(self._api.nvEncCreateBitstreamBuffer)

        self.nvEncDestroyBitstreamBuffer = CFUNCTYPE(
            c_uint32, c_void_p, c_void_p
        )(self._api.nvEncDestroyBitstreamBuffer)

        self.nvEncRegisterResource = CFUNCTYPE(
            c_uint32, c_void_p, POINTER(NV_ENC_REGISTER_RESOURCE)
        )(self._api.nvEncRegisterResource)

        self.nvEncUnregisterResource = CFUNCTYPE(
            c_uint32, c_void_p, c_void_p
        )(self._api.nvEncUnregisterResource)

        self.nvEncMapInputResource = CFUNCTYPE(
            c_uint32, c_void_p, POINTER(NV_ENC_MAP_INPUT_RESOURCE)
        )(self._api.nvEncMapInputResource)

        self.nvEncUnmapInputResource = CFUNCTYPE(
            c_uint32, c_void_p, c_void_p
        )(self._api.nvEncUnmapInputResource)

        self.nvEncEncodePicture = CFUNCTYPE(
            c_uint32, c_void_p, POINTER(NV_ENC_PIC_PARAMS)
        )(self._api.nvEncEncodePicture)

        self.nvEncLockBitstream = CFUNCTYPE(
            c_uint32, c_void_p, POINTER(NV_ENC_LOCK_BITSTREAM)
        )(self._api.nvEncLockBitstream)

        self.nvEncUnlockBitstream = CFUNCTYPE(
            c_uint32, c_void_p, c_void_p
        )(self._api.nvEncUnlockBitstream)

        self.nvEncDestroyEncoder = CFUNCTYPE(
            c_uint32, c_void_p
        )(self._api.nvEncDestroyEncoder)

        self.nvEncGetLastErrorString = CFUNCTYPE(
            ctypes.c_char_p, c_void_p
        )(self._api.nvEncGetLastErrorString)
