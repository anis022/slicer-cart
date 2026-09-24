import gzip
import struct
from math import sqrt
from pathlib import Path
from typing import Optional

HEADER_SIZE = 348

_OFFSET_PIXDIM = 76
_OFFSET_QFORM_CODE = 252
_OFFSET_SFORM_CODE = 254
_OFFSET_QUATERN = 256
_OFFSET_QOFFSET = 268
_OFFSET_SROW_X = 280
_OFFSET_SROW_Y = 296
_OFFSET_SROW_Z = 312

_AXIS_LABELS = (("R", "L"), ("A", "P"), ("S", "I"))


def read_orientation(path: Path) -> Optional[str]:
    affine = read_affine(path)
    if affine is None:
        return None
    return affine_to_orientation(affine)


def read_affine(path: Path) -> Optional[list[list[float]]]:
    raw, endian = _read_header(path)

    if struct.unpack_from(endian + "h", raw, _OFFSET_SFORM_CODE)[0] > 0:
        return [
            list(struct.unpack_from(endian + "4f", raw, offset))
            for offset in (_OFFSET_SROW_X, _OFFSET_SROW_Y, _OFFSET_SROW_Z)
        ]

    if struct.unpack_from(endian + "h", raw, _OFFSET_QFORM_CODE)[0] > 0:
        return _qform_affine(raw, endian)

    return None


def affine_to_orientation(affine: list[list[float]]) -> str:
    code = ""
    for column in range(3):
        values = [affine[row][column] for row in range(3)]
        dominant = max(range(3), key=lambda row: abs(values[row]))
        towards, away = _AXIS_LABELS[dominant]
        code += towards if values[dominant] > 0 else away
    return code


def _read_header(path: Path) -> tuple[bytes, str]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fp:
        raw = fp.read(HEADER_SIZE)

    if len(raw) < HEADER_SIZE:
        raise ValueError(
            f"File '{path.name}' is too small to contain a NIfTI-1 header; "
            "ensure it is a valid '.nii' file!"
        )

    for endian in ("<", ">"):
        if struct.unpack_from(endian + "i", raw, 0)[0] == HEADER_SIZE:
            return raw, endian

    raise ValueError(
        f"File '{path.name}' is not a NIfTI-1 file; its header does not begin "
        f"with the expected size of {HEADER_SIZE}!"
    )


def _qform_affine(raw: bytes, endian: str) -> list[list[float]]:
    b, c, d = struct.unpack_from(endian + "3f", raw, _OFFSET_QUATERN)
    offset = struct.unpack_from(endian + "3f", raw, _OFFSET_QOFFSET)
    pixdim = struct.unpack_from(endian + "8f", raw, _OFFSET_PIXDIM)

    remainder = 1.0 - (b * b + c * c + d * d)
    a = sqrt(remainder) if remainder > 0 else 0.0

    rotation = [
        [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
        [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
        [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
    ]

    qfac = -1.0 if pixdim[0] < 0 else 1.0
    scale = (pixdim[1], pixdim[2], pixdim[3] * qfac)

    return [
        [rotation[row][col] * scale[col] for col in range(3)] + [offset[row]]
        for row in range(3)
    ]
