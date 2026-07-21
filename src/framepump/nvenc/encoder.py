"""High-level NVENC encoder for OpenGL textures."""

from __future__ import annotations

import ctypes
from ctypes import POINTER, c_char_p, c_int32, c_uint32, c_void_p
from fractions import Fraction
from types import TracebackType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import moderngl

from ._session import EncodedPacket, NvencEncodeSession
from .bindings import (
    GL_TEXTURE_2D,
    NV_ENC_BUFFER_FORMAT_ABGR,
    NV_ENC_DEVICE_TYPE_OPENGL,
    NV_ENC_INPUT_IMAGE,
    NV_ENC_INPUT_RESOURCE_OPENGL_TEX,
    NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX,
    NV_ENC_REGISTER_RESOURCE,
    NV_ENC_REGISTER_RESOURCE_VER,
)
from .exceptions import EncoderNotInitialized, NvencError, TextureFormatError

__all__ = ['NvencEncoder', 'EncodedPacket']

GL_RGBA8 = 0x8058


class NvencEncoder:
    """
    High-level NVENC encoder for OpenGL textures.

    Encodes OpenGL textures directly to H.264 video using NVIDIA's hardware
    encoder without CPU memory transfers.

    Each frame is first copied into an internal ring of staging textures, so
    the caller may freely re-render into the source texture between encode()
    calls even when B-frames are enabled (NVENC reads inputs asynchronously
    while frames are buffered for reordering).

    Args:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frame rate (default: 30)
        crf: Constant quality factor (0-51, lower = better quality, default: 15)
        gop: GOP length / keyframe (IDR) interval (default: 250)
        bframes: Number of B-frames (default: 2)

    Example:
        >>> with NvencEncoder(640, 480, fps=30, crf=18) as encoder:
        ...     packets = encoder.encode(texture)
        ...     # packet.data contains H.264 NAL units
        ...     # packet.pts, packet.dts for timestamps
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: float | Fraction = 30,
        crf: int = 15,
        gop: int = 250,
        bframes: int = 2,
    ) -> None:
        self._width = width
        self._height = height
        fps = fps if isinstance(fps, Fraction) else Fraction(fps).limit_denominator(100000)
        self._closed = False
        self._session: NvencEncodeSession | None = None
        self._gl: _GLFuncs | None = None
        self._staging_ids: list[int] | None = None
        # staging texture id -> (registered resource, keep-alive GL tex struct)
        self._registered_staging: dict[int, tuple[c_void_p, NV_ENC_INPUT_RESOURCE_OPENGL_TEX]] = {}

        _check_gl_renderer_is_nvidia()

        self._session = NvencEncodeSession(
            device_type=NV_ENC_DEVICE_TYPE_OPENGL,
            device=None,
            width=width,
            height=height,
            fps=fps,
            crf=crf,
            gop=gop,
            bframes=bframes,
            open_error_hint=(
                'The current OpenGL context may be on a GPU without NVENC support\n'
                '(e.g., Intel/AMD integrated graphics).\n\n'
                'Solutions:\n'
                '  - Set __NV_PRIME_RENDER_OFFLOAD=1 to route OpenGL to the NVIDIA GPU\n'
                '  - Use DRI_PRIME=1 to select the NVIDIA GPU\n'
                '  - For headless (EGL): set DISPLAY= to use the CUDA encoder path instead'
            ),
        )

    def encode(self, texture: moderngl.Texture | int) -> list[EncodedPacket]:
        """
        Encode a frame from an OpenGL texture.

        The texture content is copied into an internal staging texture before
        submission, so the caller may modify or re-render the source texture
        immediately after this call returns.

        Args:
            texture: A moderngl.Texture object or OpenGL texture ID (int).
                     The texture must be RGBA8 format.

        Returns:
            List of EncodedPackets. Empty if the frame was buffered for
            B-frame reordering; one or more packets when output is ready.

        Note:
            Ensure OpenGL commands are complete before calling this method.
            If using moderngl, call ctx.finish() first.
        """
        if self._closed:
            raise EncoderNotInitialized('Encoder has been closed')

        if not isinstance(texture, int):
            # A larger texture would otherwise be silently cropped to the
            # encoder dimensions (raw GL ids cannot be checked).
            if texture.size != (self._width, self._height):
                raise ValueError(
                    f'Texture size {texture.size} does not match encoder '
                    f'dimensions ({self._width}, {self._height})'
                )

        texture_id = _get_texture_id(texture)
        if self._staging_ids is None:
            self._create_staging_textures()

        slot = self._session.next_submit_index % len(self._staging_ids)
        staging_id = self._staging_ids[slot]

        while self._gl.get_error() != 0:  # Drain stale errors from caller GL code
            pass
        self._gl.copy_image_sub_data(
            texture_id,
            GL_TEXTURE_2D,
            0,
            0,
            0,
            0,
            staging_id,
            GL_TEXTURE_2D,
            0,
            0,
            0,
            0,
            self._width,
            self._height,
            1,
        )
        gl_error = self._gl.get_error()
        if gl_error != 0:
            raise TextureFormatError(
                f'Copying the source texture into the staging texture failed '
                f'(GL error 0x{gl_error:04X}). The source texture must be RGBA8 '
                f'and {self._width}x{self._height}.'
            )

        registered = self._register_texture(staging_id)
        return self._session.submit(registered, self._width, self._height, self._width * 4)

    def flush(self) -> list[EncodedPacket]:
        """Flush any buffered frames from the encoder. Idempotent.

        Call this before close() to retrieve remaining packets when using
        B-frames.

        Returns:
            List of EncodedPackets for any frames still in the reorder buffer.
        """
        if self._closed or self._session is None:
            return []
        return self._session.flush()

    def close(self) -> None:
        """Release encoder resources.

        Note: Call flush() first if you need remaining buffered packets.
        The OpenGL context must still be current.
        """
        if self._closed:
            return
        self._closed = True

        if self._session is not None:
            self._session.close()
        self._registered_staging.clear()

        if self._staging_ids and self._gl is not None:
            ids = (c_uint32 * len(self._staging_ids))(*self._staging_ids)
            self._gl.delete_textures(len(self._staging_ids), ids)
            self._staging_ids = None

    def _create_staging_textures(self) -> None:
        self._gl = _GLFuncs()
        n = self._session.ring_size
        ids = (c_uint32 * n)()
        self._gl.create_textures(GL_TEXTURE_2D, n, ids)
        for staging_id in ids:
            self._gl.texture_storage_2d(staging_id, 1, GL_RGBA8, self._width, self._height)
        gl_error = self._gl.get_error()
        if gl_error != 0:
            # Delete the created names so a failed allocation leaks nothing
            self._gl.delete_textures(n, ids)
            raise NvencError(f'Failed to allocate staging textures (GL error 0x{gl_error:04X})')
        self._staging_ids = list(ids)

    def _register_texture(self, texture_id: int) -> c_void_p:
        """Register a staging texture with NVENC (once per texture)."""
        if texture_id in self._registered_staging:
            return self._registered_staging[texture_id][0]

        gl_tex_resource = NV_ENC_INPUT_RESOURCE_OPENGL_TEX()
        gl_tex_resource.texture = texture_id
        gl_tex_resource.target = GL_TEXTURE_2D

        register_params = NV_ENC_REGISTER_RESOURCE()
        register_params.version = NV_ENC_REGISTER_RESOURCE_VER
        register_params.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_OPENGL_TEX
        register_params.width = self._width
        register_params.height = self._height
        register_params.pitch = self._width * 4  # RGBA = 4 bytes per pixel
        register_params.resourceToRegister = ctypes.addressof(gl_tex_resource)
        register_params.bufferFormat = NV_ENC_BUFFER_FORMAT_ABGR
        register_params.bufferUsage = NV_ENC_INPUT_IMAGE

        registered = self._session.register_input(register_params)
        # Keep gl_tex_resource alive for the session's lifetime
        self._registered_staging[texture_id] = (registered, gl_tex_resource)
        return registered

    def __enter__(self) -> NvencEncoder:
        return self

    def __del__(self) -> None:
        if not self._closed:
            self.close()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def _get_texture_id(texture: moderngl.Texture | int) -> int:
    """Extract OpenGL texture ID from moderngl.Texture or int."""
    if isinstance(texture, int):
        return texture
    return texture.glo


def _check_gl_renderer_is_nvidia() -> None:
    """Verify the current GL context is on an NVIDIA GPU.

    NVENC with NV_ENC_DEVICE_TYPE_OPENGL requires an NVIDIA-backed context;
    on hybrid GPU systems (e.g., AMD iGPU + NVIDIA dGPU), the default context
    may be on the non-NVIDIA GPU, which causes a segfault in the NVENC driver.
    """
    try:
        gl = ctypes.cdll.LoadLibrary('libGL.so.1')
        gl.glGetString.restype = c_char_p
        renderer = (gl.glGetString(0x1F01) or b'').decode(errors='replace')
    except Exception:
        renderer = ''

    if renderer and 'nvidia' not in renderer.lower():
        raise NvencError(
            f'Current OpenGL context is on a non-NVIDIA GPU: {renderer}\n\n'
            'NVENC requires an NVIDIA-backed OpenGL context.\n\n'
            'Solutions:\n'
            '  - Set __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia\n'
            '    to route OpenGL to the NVIDIA GPU\n'
            '  - For headless (EGL): unset DISPLAY to use the CUDA encoder path'
        )


class _GLFuncs:
    """Direct-state-access OpenGL entry points for the staging-texture copies.

    Loaded via glXGetProcAddress so core-profile functions resolve on all
    driver setups; DSA (OpenGL 4.5) avoids disturbing any bound GL state
    behind the caller's back.
    """

    def __init__(self) -> None:
        lib = ctypes.CDLL('libGL.so.1')
        try:
            get_proc = lib.glXGetProcAddressARB
        except AttributeError:
            get_proc = lib.glXGetProcAddress
        get_proc.restype = c_void_p
        get_proc.argtypes = [c_char_p]

        def load(name: str, restype, *argtypes):
            addr = get_proc(name.encode())
            if not addr:
                raise NvencError(
                    f'Required OpenGL function {name} is unavailable '
                    f'(OpenGL 4.5 or newer is needed)'
                )
            return ctypes.CFUNCTYPE(restype, *argtypes)(addr)

        GLuint, GLenum, GLint, GLsizei = c_uint32, c_uint32, c_int32, c_int32
        self.create_textures = load('glCreateTextures', None, GLenum, GLsizei, POINTER(GLuint))
        self.texture_storage_2d = load(
            'glTextureStorage2D', None, GLuint, GLsizei, GLenum, GLsizei, GLsizei
        )
        self.copy_image_sub_data = load(
            'glCopyImageSubData',
            None,
            GLuint,
            GLenum,
            GLint,
            GLint,
            GLint,
            GLint,
            GLuint,
            GLenum,
            GLint,
            GLint,
            GLint,
            GLint,
            GLsizei,
            GLsizei,
            GLsizei,
        )
        self.delete_textures = load('glDeleteTextures', None, GLsizei, POINTER(GLuint))
        self.get_error = load('glGetError', GLenum)
