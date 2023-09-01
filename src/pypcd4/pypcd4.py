from __future__ import annotations

import re
import struct
from dataclasses import dataclass, fields
from pathlib import Path
from typing import BinaryIO, Iterable, TextIO

import lzf
import numpy as np

NUMPY_TYPE_TO_PCD_TYPE: dict[
    type[np.floating | np.integer],
    tuple[str, int],
] = {
    np.uint8: ("U", 1),
    np.uint16: ("U", 2),
    np.uint32: ("U", 4),
    np.uint64: ("U", 8),
    np.int16: ("I", 2),
    np.int32: ("I", 4),
    np.int64: ("I", 8),
    np.float32: ("F", 4),
    np.float64: ("F", 8),
}

PCD_TYPE_TO_NUMPY_TYPE: dict[
    tuple[str, int],
    np.dtype,
] = {
    ("F", 4): np.dtype("float32"),
    ("F", 8): np.dtype("float64"),
    ("U", 1): np.dtype("uint8"),
    ("U", 2): np.dtype("uint16"),
    ("U", 4): np.dtype("uint32"),
    ("U", 8): np.dtype("uint64"),
    ("I", 2): np.dtype("int16"),
    ("I", 4): np.dtype("int32"),
    ("I", 8): np.dtype("int64"),
}


@dataclass()
class MetaData:
    version: str = "0.7"
    fields: Iterable[str] | None = None
    size: Iterable[int] | None = None
    type: Iterable[str] | None = None
    count: Iterable[int] | None = None
    width: int | None = None
    height: int = 1
    viewpoint: Iterable[float] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    points: int | None = None
    data: str = "binary_compressed"

    def validate(self) -> bool:
        for field in fields(self):
            if getattr(self, field.name) is None:
                return False

        if not (len(self.fields) == len(self.size) == len(self.type) == len(self.count)):
            return False

        return True

    def build_dtype(self) -> np.dtype:
        field_names: list[str] = []
        type_names: list[np.dtype] = []

        for i, field in enumerate(self.fields):
            np_type = PCD_TYPE_TO_NUMPY_TYPE[(self.type[i], self.size[i])]

            if (count := self.count[i]) == 1:
                field_names.append(self.fields[i])
                type_names.append(np_type)
            else:
                field_names.extend([f"{field}_{i:04d}" for i in range(count)])
                type_names.extend([np_type] * count)

        return np.dtype([x for x in zip(field_names, type_names)])

    @staticmethod
    def parse_header(lines: list[str]) -> MetaData:
        """
        Parse a header in accordance with the PCD (Point Cloud Data) file format.
        See https://pointclouds.org/documentation/tutorials/pcd_file_format.html.
        """

        metadata = MetaData()

        for line in lines:
            if line.startswith("#") or len(line) < 2:
                continue

            line = line.replace("_", "s", 1)
            line = line.replace("_", "m", 1)

            if match := re.match(r"(\w+)\s+([\w\s\.]+)", line):
                key, value = match.group(1).lower(), match.group(2)

                if key == "version":
                    metadata.version = value
                elif key == "fields":
                    metadata.fields = value.split()
                elif key == "size":
                    metadata.size = [int(v) for v in value.split()]
                elif key == "type":
                    metadata.type = value.split()
                elif key == "count":
                    metadata.count = [int(v) for v in value.split()]
                elif key == "width":
                    metadata.width = int(value)
                elif key == "height":
                    metadata.height = int(value)
                elif key == "viewpoint":
                    metadata.viewpoint = [float(v) for v in value.split()]
                elif key == "points":
                    metadata.points = int(value)
                elif key == "data":
                    metadata.data = value.strip().lower()

        return metadata

    def compose_header(self) -> str:
        header: list[str] = []

        header.append(f"VERSION {self.version}")
        header.append(f"FIELDS {' '.join(self.fields)}")
        header.append(f"SIZE {' '.join([str(v) for v in self.size])}")
        header.append(f"TYPE {' '.join(self.type)}")
        header.append(f"COUNT {' '.join([str(v) for v in self.count])}")
        header.append(f"WIDTH {self.width}")
        header.append(f"HEIGHT {self.height}")
        header.append(f"VIEWPOINT {' '.join([str(v) for v in self.viewpoint])}")
        header.append(f"POINTS {self.points}")
        header.append(f"DATA {self.data}")

        return "\n".join(header) + "\n"


def _parse_pc_data(fp: TextIO | BinaryIO, metadata: MetaData) -> np.ndarray:
    dtype = metadata.build_dtype()

    if metadata.data == "ascii":
        pc_data = np.loadtxt(fp, dtype=dtype, delimiter=" ")
    elif metadata.data == "binary":
        pc_data = np.frombuffer(fp.read(metadata.points * dtype.itemsize), dtype=dtype)
    elif metadata.data == "binary_compressed" or metadata.data == "binaryscompressed":
        compressed_size, uncompressed_size = struct.unpack(
            "II", fp.read(8)  # <--- struct.calcsize("II")
        )

        buffer = lzf.decompress(fp.read(compressed_size), uncompressed_size)
        if (actual_size := len(buffer)) != uncompressed_size:
            raise RuntimeError(
                f"Failed to decompress data. "
                f"Expected decompressed file size is {uncompressed_size}, but got {actual_size}"
            )

        offset = 0
        pc_data = np.zeros(metadata.width, dtype=dtype)
        for dti in range(len(dtype)):
            dt: np.dtype = dtype[dti]
            bytes = dt.itemsize * metadata.width
            pc_data[dtype.names[dti]] = np.frombuffer(buffer[offset : (offset + bytes)], dtype=dt)
            offset += bytes
    else:
        raise RuntimeError(
            f"DATA field is neither 'ascii' or 'binary' or 'binary_compressed', "
            f"but got {metadata.data}"
        )

    return pc_data


def _compose_pc_data(points: np.ndarray, metadata: MetaData) -> np.ndarray:
    return np.rec.fromarrays(
        [points[:, i] for i in range(len(metadata.fields))],
        dtype=metadata.build_dtype(),
    )


class PointCloud:
    def __init__(self, metadata: MetaData, pc_data: np.ndarray) -> None:
        self.metadata = metadata
        self.pc_data = pc_data

        if not self.metadata.validate():
            raise RuntimeError(f"Got broken metadata. Check if the data is valid. {self.metadata}")

    @staticmethod
    def from_fileobj(fp: TextIO | BinaryIO) -> PointCloud:
        lines: list[str] = []
        while True:
            line = fp.readline().strip()
            lines.append(line.decode(encoding="utf-8") if isinstance(line, bytes) else line)

            if lines[-1].startswith("DATA"):
                break

        metadata = MetaData.parse_header(lines)
        pc_data = _parse_pc_data(fp, metadata)

        return PointCloud(metadata, pc_data)

    @staticmethod
    def from_path(path: str | Path) -> PointCloud:
        with open(path, mode="rb") as fp:
            return PointCloud.from_fileobj(fp)

    @staticmethod
    def from_points(
        points: np.ndarray,
        fields: Iterable[str],
        types: Iterable[type[np.floating | np.integer]],
        count: Iterable[int] | None = None,
    ) -> PointCloud:
        if count is None:
            count = [1] * len(tuple(fields))

        type = [NUMPY_TYPE_TO_PCD_TYPE[dtype][0] for dtype in types]
        size = [NUMPY_TYPE_TO_PCD_TYPE[dtype][1] for dtype in types]

        metadata = MetaData(
            fields=fields,
            size=size,
            type=type,
            count=count,
            width=len(points),
            points=len(points),
        )

        return PointCloud(metadata, _compose_pc_data(points, metadata))

    @staticmethod
    def from_xyz_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z")
        types = (np.float32, np.float32, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzi_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "intensity")
        types = (np.float32, np.float32, np.float32, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzl_points(
        points: np.ndarray, label_type: type[np.floating | np.integer] = np.float32
    ) -> PointCloud:
        fields = ("x", "y", "z", "label")
        types = (np.float32, np.float32, np.float32, label_type)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzrgb_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "rgb")
        types = (np.float32, np.float32, np.float32, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzrgbl_points(
        points: np.ndarray, label_type: type[np.floating | np.integer] = np.float32
    ) -> PointCloud:
        fields = ("x", "y", "z", "rgb", "label")
        types = (np.float32, np.float32, np.float32, np.float32, label_type)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzil_points(
        points: np.ndarray, label_type: type[np.floating | np.integer] = np.float32
    ) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "label")
        types = (np.float32, np.float32, np.float32, np.float32, label_type)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzirgb_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "rgb")
        types = (np.float32, np.float32, np.float32, np.float32, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzirgbl_points(
        points: np.ndarray, label_type: type[np.floating | np.integer] = np.float32
    ) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "rgb", "label")
        types = (np.float32, np.float32, np.float32, np.float32, np.float32, label_type)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzt_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "sec", "nsec")
        types = (np.float32, np.float32, np.float32, np.uint32, np.uint32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzir_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "ring")
        types = (np.float32, np.float32, np.float32, np.float32, np.uint16)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzirt_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "ring", "time")
        types = (np.float32, np.float32, np.float32, np.float32, np.uint16, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzit_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "timestamp")
        types = (np.float32, np.float32, np.float32, np.float32, np.float64)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzis_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "stamp")
        types = (np.float32, np.float32, np.float32, np.float32, np.float64)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzisc_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "stamp", "classification")
        types = (np.float32, np.float32, np.float32, np.float32, np.float64, np.uint8)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzrgbs_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "rgb", "stamp")
        types = (np.float32, np.float32, np.float32, np.float32, np.float64)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzirgbs_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "rgb", "stamp")
        types = (
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float64,
        )

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzirgbsc_points(points: np.ndarray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "rgb", "stamp", "classification")
        types = (
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float64,
            np.uint8,
        )

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyziradt_points(points: np.ndarray) -> PointCloud:
        fields = (
            "x",
            "y",
            "z",
            "intensity",
            "ring",
            "azimuth",
            "distance",
            "return_type",
            "time_stamp",
        )
        types = (
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.uint16,
            np.float32,
            np.float32,
            np.uint8,
            np.float64,
        )

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_ouster_points(points: np.ndarray) -> PointCloud:
        fields = (
            "x",
            "y",
            "z",
            "intensity",
            "t",
            "reflectivity",
            "ring",
            "ambient",
            "range",
        )
        types = (
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.float32,
            np.uint16,
            np.uint8,
            np.uint16,
            np.uint32,
        )

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def encode_rgb(rgb: np.ndarray) -> np.ndarray:
        """
        Encode Nx3 uint8 array with RGB values to
        Nx1 float32 array with bit-packed RGB
        """

        rgb = rgb.astype(np.uint32)

        return np.array((rgb[:, 0] << 16) | (rgb[:, 1] << 8) | (rgb[:, 2] << 0), dtype=np.uint32)

    @staticmethod
    def decode_rgb(rgb: np.ndarray) -> np.ndarray:
        """
        Decode Nx1 float32 array with bit-packed RGB to
        Nx3 uint8 array with RGB values
        """

        rgb = rgb.copy().astype(np.uint32)
        r = np.asarray((rgb >> 16) & 255, dtype=np.uint8)[:, None]
        g = np.asarray((rgb >> 8) & 255, dtype=np.uint8)[:, None]
        b = np.asarray(rgb & 255, dtype=np.uint8)[:, None]

        return np.hstack((r, g, b))

    @property
    def fields(self) -> Iterable[str] | None:
        return self.metadata.fields

    @property
    def type(self) -> Iterable[str] | None:
        return self.metadata.type

    @property
    def size(self) -> Iterable[int] | None:
        return self.metadata.size

    def numpy(self, fields: Iterable[str] | None = None) -> np.ndarray:
        """Convert to (N, M) numpy.ndarray points"""

        if self.fields is None:
            raise ValueError("No fields")

        if fields is None:
            fields = self.fields

        _stack = [self.pc_data[field] for field in self.fields]

        return np.vstack(_stack).T

    def save(self, path: str | Path) -> None:
        self.metadata.data = "binary_compressed"  # only support binary_compressed

        with open(path, mode="wb") as fp:
            header = self.metadata.compose_header().encode()
            fp.write(header)

            uncompresseds = [
                np.ascontiguousarray(self.pc_data[field]).tobytes()
                for field in self.pc_data.dtype.names
            ]
            uncompressed = b"".join(uncompresseds)

            if (compressed := lzf.compress(uncompressed)) is None:
                compressed = uncompressed

            fp.write(struct.pack("II", len(compressed), len(uncompressed)))
            fp.write(compressed)
