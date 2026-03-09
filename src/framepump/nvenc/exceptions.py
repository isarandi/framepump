"""NVENC-related exceptions."""

# NVENC status codes from nvEncodeAPI.h
# https://github.com/NVIDIA/video-sdk-samples/blob/master/Samples/NvCodec/NvEncoder/nvEncodeAPI.h
NVENC_STATUS_CODES = {
    0: 'NV_ENC_SUCCESS',
    1: 'NV_ENC_ERR_NO_ENCODE_DEVICE',
    2: 'NV_ENC_ERR_UNSUPPORTED_DEVICE',
    3: 'NV_ENC_ERR_INVALID_ENCODERDEVICE',
    4: 'NV_ENC_ERR_INVALID_DEVICE',
    5: 'NV_ENC_ERR_DEVICE_NOT_EXIST',
    6: 'NV_ENC_ERR_INVALID_PTR',
    7: 'NV_ENC_ERR_INVALID_EVENT',
    8: 'NV_ENC_ERR_INVALID_PARAM',
    9: 'NV_ENC_ERR_INVALID_CALL',
    10: 'NV_ENC_ERR_OUT_OF_MEMORY',
    11: 'NV_ENC_ERR_ENCODER_NOT_INITIALIZED',
    12: 'NV_ENC_ERR_UNSUPPORTED_PARAM',
    13: 'NV_ENC_ERR_LOCK_BUSY',
    14: 'NV_ENC_ERR_NOT_ENOUGH_BUFFER',
    15: 'NV_ENC_ERR_INVALID_VERSION',
    16: 'NV_ENC_ERR_MAP_FAILED',
    17: 'NV_ENC_ERR_NEED_MORE_INPUT',
    18: 'NV_ENC_ERR_ENCODER_BUSY',
    19: 'NV_ENC_ERR_EVENT_NOT_REGISTERED',
    20: 'NV_ENC_ERR_GENERIC',
    21: 'NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY',
    22: 'NV_ENC_ERR_UNIMPLEMENTED',
    23: 'NV_ENC_ERR_RESOURCE_REGISTER_FAILED',
    24: 'NV_ENC_ERR_RESOURCE_NOT_REGISTERED',
    25: 'NV_ENC_ERR_RESOURCE_NOT_MAPPED',
}


def nvenc_status_message(status: int, context: str = '') -> str:
    """Convert NVENC status code to a human-readable message."""
    name = NVENC_STATUS_CODES.get(status, f'UNKNOWN_ERROR_{status}')

    # Add helpful hints for common errors
    if status == 8:  # NV_ENC_ERR_INVALID_PARAM
        hint = ' (check resolution, frame rate, or encoding parameters)'
    elif status == 1:  # NV_ENC_ERR_NO_ENCODE_DEVICE
        hint = ' (no NVIDIA GPU with NVENC support found)'
    elif status == 10:  # NV_ENC_ERR_OUT_OF_MEMORY
        hint = ' (GPU memory exhausted)'
    elif status == 15:  # NV_ENC_ERR_INVALID_VERSION
        hint = ' (driver/SDK version mismatch)'
    else:
        hint = ''

    prefix = f'{context}: ' if context else ''
    return f'{prefix}{name}{hint}'


class NvencError(Exception):
    """Base exception for NVENC errors."""
    pass


class NvencNotAvailable(NvencError):
    """NVENC library could not be loaded."""
    pass


class TextureFormatError(NvencError):
    """Texture format is not compatible with NVENC."""
    pass


class EncoderNotInitialized(NvencError):
    """Encoder was not properly initialized."""
    pass