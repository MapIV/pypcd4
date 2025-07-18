from __future__ import annotations

import random
import re
import string
import struct
from enum import Enum
from io import BufferedReader
from pathlib import Path
from typing import TYPE_CHECKING, BinaryIO, List, Literal, Optional, Sequence, Tuple, Union, cast

import lzf  # type: ignore
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, NonNegativeInt, PositiveInt

from .pointcloud2 import (
    NPTYPE_TO_PFTYPE,
    build_dtype_from_msg,
    sensor_msgs__msg__PointCloud2,
)

if TYPE_CHECKING:
    from sensor_msgs.msg import PointCloud2  # type: ignore
    from std_msgs.msg import Header  # type: ignore

PathLike = Union[str, Path]

MetaDataVersion = Literal[".7", "0.7"]
MetaDataViewPoint = Tuple[float, float, float, float, float, float, float]

NUMPY_TYPE_TO_PCD_TYPE: dict[
    npt.DTypeLike,
    Tuple[str, int],
] = {
    np.dtype("uint8"): ("U", 1),
    np.dtype("uint16"): ("U", 2),
    np.dtype("uint32"): ("U", 4),
    np.dtype("uint64"): ("U", 8),
    np.dtype("int8"): ("I", 1),
    np.dtype("int16"): ("I", 2),
    np.dtype("int32"): ("I", 4),
    np.dtype("int64"): ("I", 8),
    np.dtype("float32"): ("F", 4),
    np.dtype("float64"): ("F", 8),
}

PCD_TYPE_TO_NUMPY_TYPE: dict[
    Tuple[str, int],
    npt.DTypeLike,
] = {
    ("F", 4): np.float32,
    ("F", 8): np.float64,
    ("U", 1): np.uint8,
    ("U", 2): np.uint16,
    ("U", 4): np.uint32,
    ("U", 8): np.uint64,
    ("I", 1): np.int8,
    ("I", 2): np.int16,
    ("I", 4): np.int32,
    ("I", 8): np.int64,
}

HEADER_PATTERN = re.compile(r"(\w+)\s+([\w\s\.\-?\d+\.?\d*]+)")


class Encoding(str, Enum):
    ASCII = "ascii"
    BINARY = "binary"
    BINARY_COMPRESSED = "binary_compressed"
    BINARYSCOMPRESSED = "binaryscompressed"


class MetaData(BaseModel):
    fields: Tuple[str, ...]
    size: Tuple[PositiveInt, ...]
    type: Tuple[Literal["F", "U", "I"], ...]
    count: Tuple[PositiveInt, ...]
    points: NonNegativeInt
    width: NonNegativeInt
    height: NonNegativeInt = 1
    version: MetaDataVersion = "0.7"
    viewpoint: MetaDataViewPoint = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    data: Encoding = Encoding.BINARY_COMPRESSED

    @staticmethod
    def parse_header(lines: List[str]) -> MetaData:
        """
        Parse a header in accordance with the PCD (Point Cloud Data) file format.
        See https://pointclouds.org/documentation/tutorials/pcd_file_format.html.

        Args:
            lines (List[str]): A list of lines of the header

        Raises:
            pydantic.ValidationError: If the header is invalid

        Returns:
            MetaData: Parsed MetaData object
        """

        _header = {}
        for line in lines:
            if line.startswith("#") or len(line) < 2:  # noqa: PLR2004
                continue

            if (match := re.match(HEADER_PATTERN, line)) is None:
                continue

            value = match.group(2).split()
            if (key := match.group(1).lower()) in ["version", "data"]:
                _header[key] = value[0]
            elif key in ["width", "height", "points"]:
                _header[key] = int(value[0])
            elif key in ["fields"]:
                _header[key] = tuple(
                    [
                        "#$%&~~"
                        + "".join(random.choices(string.ascii_letters + string.digits, k=6))
                        if s == "_"
                        else s
                        for s in value
                    ]
                )
            elif key in ["type"]:
                _header[key] = tuple(value)
            elif key in ["size", "count"]:
                _header[key] = tuple(int(v) for v in value)
            elif key in ["viewpoint"]:
                _header[key] = tuple(float(v) for v in value)
            else:
                pass
        try:
            metadata = MetaData.model_validate(_header)  # type: ignore[attr-defined]
        except AttributeError:
            metadata = MetaData.parse_obj(_header)  # type: ignore[attr-defined]

        return metadata

    def compose_header(self) -> str:
        header: List[str] = []

        header.append(f"VERSION {self.version}")
        header.append(
            f"FIELDS {' '.join(['_' if s.startswith('#$%&~~') else s for s in self.fields])}"
        )
        header.append(f"SIZE {' '.join([str(v) for v in self.size])}")
        header.append(f"TYPE {' '.join(self.type)}")
        header.append(f"COUNT {' '.join([str(v) for v in self.count])}")
        header.append(f"WIDTH {self.width}")
        header.append(f"HEIGHT {self.height}")
        header.append(f"VIEWPOINT {' '.join([str(v) for v in self.viewpoint])}")
        header.append(f"POINTS {self.points}")
        header.append(f"DATA {self.data.value}")

        return "\n".join(header) + "\n"

    def build_dtype(self) -> np.dtype[np.void]:
        field_names: List[str] = []
        np_types: List[npt.DTypeLike] = []

        for i, field in enumerate(self.fields):
            np_type: npt.DTypeLike = np.dtype(PCD_TYPE_TO_NUMPY_TYPE[(self.type[i], self.size[i])])

            if (count := self.count[i]) == 1:
                field_names.append(field)
                np_types.append(np_type)
            else:
                field_names.extend([f"{field}__{i:04d}" for i in range(count)])
                np_types.extend([np_type] * count)

        return np.dtype([x for x in zip(field_names, np_types)])


def _parse_pc_data(fp: BufferedReader, metadata: MetaData) -> npt.NDArray:
    dtype = metadata.build_dtype()

    if metadata.points > 0:
        if metadata.data == Encoding.ASCII:
            pc_data = np.loadtxt(fp, dtype, delimiter=" ")
        elif metadata.data == Encoding.BINARY:
            buffer = fp.read(metadata.points * dtype.itemsize)
            pc_data = np.frombuffer(buffer, dtype)
        else:
            compressed_size, uncompressed_size = struct.unpack("II", fp.read(8))  # type: ignore

            buffer = lzf.decompress(fp.read(compressed_size), uncompressed_size)
            if (actual_size := len(buffer)) != uncompressed_size:
                raise RuntimeError(
                    f"Failed to decompress data. "
                    f"Expected decompressed file size is {uncompressed_size}, but got {actual_size}"
                )

            offset = 0
            pc_data = np.zeros(metadata.points, dtype=dtype)
            for name in dtype.names:  # type: ignore
                dt: np.dtype = dtype[name]
                bytes = dt.itemsize * metadata.points
                pc_data[name] = np.frombuffer(buffer[offset : (offset + bytes)], dtype=dt)
                offset += bytes
    else:
        pc_data = np.empty((0, len(metadata.fields)), dtype)

    return pc_data


def _compose_pc_data(
    points: Union[npt.NDArray, Sequence[npt.NDArray]], metadata: MetaData
) -> npt.NDArray:
    arrays = tuple(points.T) if isinstance(points, np.ndarray) else points

    return np.rec.fromarrays(arrays, dtype=metadata.build_dtype())


class PointCloud:
    def __init__(self, metadata: MetaData, pc_data: npt.NDArray) -> None:
        self.metadata = metadata
        self.pc_data = pc_data

    @staticmethod
    def from_fileobj(fp: BufferedReader) -> PointCloud:
        lines: List[str] = []
        for bline in fp:
            if (line := bline.decode(encoding="utf-8").strip()).startswith("#") or not line:
                continue

            lines.append(line)

            if line.startswith("DATA") or len(lines) >= 10:
                break

        metadata = MetaData.parse_header(lines)
        pc_data = _parse_pc_data(fp, metadata)

        fp.seek(0)

        return PointCloud(metadata, pc_data)

    @staticmethod
    def from_path(path: PathLike) -> PointCloud:
        """Create PointCloud from file path

        Args:
            path (PathLike): File path

        Raises:
            FileNotFoundError: Raise if `path` does not exist.

        Returns:
            PointCloud: PointCloud object

        >>> PointCloud.from_path("test.pcd")
        PointCloud(
            fields=('x', 'y', 'z') size=(4, 4, 4) type=('F', 'F', 'F') count=(1, 1, 1) points=10000
            width=10000 height=1 version='0.7' viewpoint=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            data=<Encoding.BINARY_COMPRESSED: 'binary_compressed'>
        )
        """

        with open(path, mode="rb") as fp:
            return PointCloud.from_fileobj(fp)

    @staticmethod
    def from_points(
        points: Union[npt.NDArray, List[npt.NDArray], Tuple[npt.NDArray, ...]],
        fields: Sequence[str],
        types: Sequence[npt.DTypeLike],
        count: Optional[Sequence[int]] = None,
    ) -> PointCloud:
        """Create PointCloud from 2D numpy array or sequence of 1D numpy arrays of each field

        Args:
            points (Union[npt.NDArray, List[npt.NDArray], Tuple[npt.NDArray, ...]]): Numpy array
            fields (Sequence[str]): Field names for each numpy array or sequence of numpy arrays
            types (Sequence[npt.DTypeLike]): Numpy type for each numpy array or
                sequence of numpy arrays
            count (Optional[Sequence[int]], optional): Number of values for each numpy array or
                sequence of numpy arrays. If None, set to 1 for all fields. Defaults to None.

        Raises:
            TypeError: Raise if `points` type is neither `npt.NDArray` nor `Sequence[npt.NDArray]`.

        Returns:
            PointCloud: PointCloud object

        >>> PointCloud.from_points(
                np.array([[1, 2, 3], [4, 5, 6]]),
                ("x", "y", "z"),
                (np.float32, np.float32, np.float32)
            )
        PointCloud(
            fields=('x', 'y', 'z') size=(4, 4, 4) type=('F', 'F', 'F') count=(1, 1, 1) points=2
            width=2 height=1 version='0.7' viewpoint=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            data=<Encoding.BINARY_COMPRESSED: 'binary_compressed'>
        )

        >>> PointCloud.from_points(
                [
                    np.array([1, 2, 3]),
                    np.array([4, 5, 6])
                ],
                ("x", "y", "z"),
                (np.float32, np.float32, np.float32)
            )
        PointCloud(
            fields=('x', 'y', 'z') size=(4, 4, 4) type=('F', 'F', 'F') count=(1, 1, 1) points=2
            width=2 height=1 version='0.7' viewpoint=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            data=<Encoding.BINARY_COMPRESSED: 'binary_compressed'>
        )
        """

        if not isinstance(points, (np.ndarray, list, tuple)):
            raise TypeError(
                "`points` must be a `np.ndarray`, `List[np.ndarray]` or `Tuple[np.ndarray]` type, "
                f"but got `{type(points).__name__}` type"
            )

        if len(fields) != len(types):
            raise ValueError(
                f"`fields` and `types` must have the same length, "
                f"but got {len(fields)} fields and {len(types)} types"
            )

        num_fields = points.shape[1] if isinstance(points, np.ndarray) else len(points)
        if num_fields != len(fields):
            raise ValueError(
                f"`points` must have {len(fields)} fields {fields}, but got {num_fields} fields"
            )

        if count is None:
            count = [1] * len(tuple(fields))

        pcd_types, sizes = [], []
        for type_ in types:
            t, s = NUMPY_TYPE_TO_PCD_TYPE[np.dtype(type_)]
            pcd_types.append(t)
            sizes.append(s)

        num_points = len(points) if isinstance(points, np.ndarray) else len(points[0])

        try:
            metadata = MetaData.model_validate(  # type: ignore[attr-defined]
                {
                    "fields": fields,
                    "size": sizes,
                    "type": pcd_types,
                    "count": count,
                    "width": num_points,
                    "points": num_points,
                }
            )
        except AttributeError:
            metadata = MetaData.parse_obj(  # type: ignore[attr-defined]
                {
                    "fields": fields,
                    "size": sizes,
                    "type": pcd_types,
                    "count": count,
                    "width": num_points,
                    "points": num_points,
                    "height": 1,
                    "version": "0.7",
                    "viewpoint": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                    "data": Encoding.BINARY_COMPRESSED,
                }
            )

        return PointCloud(metadata, _compose_pc_data(points, metadata))

    @staticmethod
    def from_xyz_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z")
        types = (np.float32, np.float32, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzi_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "intensity")
        types = (np.float32, np.float32, np.float32, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzl_points(points: npt.NDArray, label_type: npt.DTypeLike = np.float32) -> PointCloud:
        fields = ("x", "y", "z", "label")
        types = (np.float32, np.float32, np.float32, label_type)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzrgb_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "rgb")
        types = (np.float32, np.float32, np.float32, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzrgbl_points(
        points: npt.NDArray, label_type: npt.DTypeLike = np.float32
    ) -> PointCloud:
        fields = ("x", "y", "z", "rgb", "label")
        types = (np.float32, np.float32, np.float32, np.float32, label_type)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzil_points(
        points: npt.NDArray, label_type: npt.DTypeLike = np.float32
    ) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "label")
        types = (np.float32, np.float32, np.float32, np.float32, label_type)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzirgb_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "rgb")
        types = (np.float32, np.float32, np.float32, np.float32, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzirgbl_points(
        points: npt.NDArray, label_type: npt.DTypeLike = np.float32
    ) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "rgb", "label")
        types = (np.float32, np.float32, np.float32, np.float32, np.float32, label_type)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzt_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "sec", "nsec")
        types = (np.float32, np.float32, np.float32, np.uint32, np.uint32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzir_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "ring")
        types = (np.float32, np.float32, np.float32, np.float32, np.uint16)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzirt_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "ring", "time")
        types = (np.float32, np.float32, np.float32, np.float32, np.uint16, np.float32)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzit_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "timestamp")
        types = (np.float32, np.float32, np.float32, np.float32, np.float64)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzis_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "stamp")
        types = (np.float32, np.float32, np.float32, np.float32, np.float64)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzisc_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "intensity", "stamp", "classification")
        types = (np.float32, np.float32, np.float32, np.float32, np.float64, np.uint8)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzrgbs_points(points: npt.NDArray) -> PointCloud:
        fields = ("x", "y", "z", "rgb", "stamp")
        types = (np.float32, np.float32, np.float32, np.float32, np.float64)

        return PointCloud.from_points(points, fields, types)

    @staticmethod
    def from_xyzirgbs_points(points: npt.NDArray) -> PointCloud:
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
    def from_xyzirgbsc_points(points: npt.NDArray) -> PointCloud:
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
    def from_xyziradt_points(points: npt.NDArray) -> PointCloud:
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
    def from_ouster_points(points: npt.NDArray) -> PointCloud:
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
    def from_list(pcs: List[PointCloud]) -> PointCloud:
        if not all(pc.fields == pcs[0].fields and pc.types == pcs[0].types for pc in pcs):
            raise ValueError("from_list: All PointClouds must have the same fields and types")
        return PointCloud.from_points(
            np.concatenate([pc.numpy() for pc in pcs]), pcs[0].fields, pcs[0].types
        )

    @staticmethod
    def from_msg(msg: sensor_msgs__msg__PointCloud2) -> PointCloud:
        """Create a PointCloud from a ROS/ROS 2 sensor_msgs.msg.PointCloud2 message

        Args:
            msg: The ROS/ROS 2 sensor_msgs/PointCloud2 message

        Returns:
            The PointCloud
        """

        pc_data = np.frombuffer(msg.data, build_dtype_from_msg(msg))
        metadata = MetaData.model_validate(
            {
                "fields": pc_data.dtype.names,
                "size": [
                    NUMPY_TYPE_TO_PCD_TYPE[pc_data[name].dtype][1]
                    for name in pc_data.dtype.names  # type: ignore[union-attr]
                ],
                "type": [
                    NUMPY_TYPE_TO_PCD_TYPE[pc_data[name].dtype][0]
                    for name in pc_data.dtype.names  # type: ignore[union-attr]
                ],
                "count": [1] * len(pc_data.dtype.names),  # type: ignore[arg-type]
                "points": len(pc_data),
                "width": msg.width,
                "height": msg.height,
                "version": "0.7",
                "viewpoint": (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0),
                "data": Encoding.BINARY,
            }
        )

        return PointCloud(metadata, pc_data)

    def to_msg(self, header: Optional["Header"] = None) -> "PointCloud2":
        """Create a ROS/ROS 2 sensor_msgs.msg.PointCloud2 message from the PointCloud

        Args:
            header: The header to use for the message.
                    If None, a default header will be used. Default: None.

        Returns:
            The ROS/ROS 2 sensor_msgs/PointCloud2 message
        """
        ROS_MSG_AVAILABLE = False
        try:
            from sensor_msgs.msg import PointCloud2, PointField  # type: ignore
            from std_msgs.msg import Header  # type: ignore

            ROS_MSG_AVAILABLE = True
            try:
                from builtin_interfaces.msg import Time  # ROS2  # type: ignore
            except ImportError:
                from std_msgs.msg import Time  # ROS1  # type: ignore
        except ImportError:
            pass

        if not ROS_MSG_AVAILABLE:
            try:
                # Fallback to rosbags
                from rosbags.typesys.stores.latest import (  # type: ignore
                    builtin_interfaces__msg__Time as Time,
                )
                from rosbags.typesys.stores.latest import (  # type: ignore
                    sensor_msgs__msg__PointCloud2 as PointCloud2,
                )
                from rosbags.typesys.stores.latest import (  # type: ignore
                    sensor_msgs__msg__PointField as PointField,
                )
                from rosbags.typesys.stores.latest import (  # type: ignore
                    std_msgs__msg__Header as Header,
                )

                ROS_MSG_AVAILABLE = True
            except ImportError:
                pass

        if not ROS_MSG_AVAILABLE:
            raise ImportError(
                "The PointCloud.to_msg() method requires ROS `sensor_msgs` or the `rosbags` pkg.\n"
                "Please install either:\n"
                "  - ROS/ROS2 (check official documentation), or\n"
                "  - rosbags via `pip install rosbags`."
            )
        fields = []
        itemsize = 0
        row_step = 0
        offset = 0
        for field, type_, count in zip(self.fields, self.types, self.counts):
            type_ = np.dtype(type_)

            itemsize += type_.itemsize
            row_step += type_.itemsize * self.points

            fields.append(
                PointField(
                    name=field,
                    offset=offset,
                    datatype=NPTYPE_TO_PFTYPE[type_],
                    count=count,
                )
            )
            offset += type_.itemsize * count

        data = self.pc_data.reshape(-1).view(np.uint8)

        if header is None:
            try:
                header = Header(stamp=Time(sec=0, nanosec=0), frame_id="")
            except (TypeError, AttributeError):
                header = Header(stamp=Time(0), frame_id="", seq=0)

        msg = PointCloud2(
            header=header,
            height=1,
            width=self.points,
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=itemsize,
            row_step=len(data),
            data=data,
        )

        return msg

    @staticmethod
    def encode_rgb(rgb: npt.NDArray | list[npt.NDArray]) -> npt.NDArray:
        """
        Encode Nx3 uint8 array with RGB values to
        Nx1 float32 array with bit-packed RGB
        """

        if isinstance(rgb, np.ndarray):
            if rgb.ndim == 1:
                rgb = rgb[:, None]
        else:
            rgb = np.hstack([v[:, None] if v.ndim == 1 else v for v in rgb])

        rgb_u32 = rgb.astype(np.uint32)
        rgb_u32 = np.array(
            (rgb_u32[:, 0] << 16) | (rgb_u32[:, 1] << 8) | (rgb_u32[:, 2] << 0),
            dtype=np.uint32,
        )
        rgb_u32.dtype = np.float32  # type: ignore

        return rgb_u32

    @staticmethod
    def decode_rgb(rgb: npt.NDArray) -> npt.NDArray:
        """
        Decode Nx1 float32 array with bit-packed RGB to
        Nx3 uint8 array with RGB values
        """

        rgb = rgb.copy()
        rgb.dtype = np.uint32  # type: ignore

        r = np.asarray((rgb >> 16) & 255, dtype=np.uint8).reshape(-1, 1)
        g = np.asarray((rgb >> 8) & 255, dtype=np.uint8).reshape(-1, 1)
        b = np.asarray(rgb & 255, dtype=np.uint8).reshape(-1, 1)

        return np.hstack((r, g, b)).reshape(-1, 3)

    @property
    def fields(self) -> Tuple[str, ...]:
        """
        Returns fields of the point cloud

        Returns:
            Tuple[str, ...]: Tuple of field names for each field

        Note:
            If the field count is not 1, the field names will be suffixed with __0000, __0001, etc.
            If you don't want this behavior, use the `pc.metadata.fields` instead.

        >>> pc1.metadata.count
        (1, 1, 1)

        >>> pc1.fields
        ("x", "y", "z")

        >>> pc2.metadata.count
        (1, 1, 2)

        >>> pc2.fields
        ("x", "y", "z__0000", "z__0001")

        >>> pc2.metadata.fields
        ("x", "y", "z")
        """

        fields = []
        for field, count in zip(self.metadata.fields, self.metadata.count):
            if count == 1:
                fields.append(field)
            else:
                fields.extend(f"{field}__{c:04d}" for c in range(count))

        return tuple(fields)

    @property
    def types(self) -> Tuple[npt.DTypeLike, ...]:
        """Returns types of the point cloud

        Returns:
            Tuple[npt.DTypeLike, ...]: Tuple of numpy types for each field

        Note:
            If the field count is not 1, the type will be repeated for the number of fields

        >>> pc1.metadata.count
        (1, 1, 1)

        >>> pc1.types
        (np.float32, np.int32, np.float32)

        >>> pc2.metadata.count
        (1, 2, 1)

        >>> pc2.types
        (np.float32, np.int32, np.int32, np.float32)
        """

        types = []
        for ts, count in zip(zip(self.metadata.type, self.metadata.size), self.metadata.count):
            if count == 1:
                types.append(PCD_TYPE_TO_NUMPY_TYPE[ts])
            else:
                types.extend(PCD_TYPE_TO_NUMPY_TYPE[ts] for _ in range(count))

        return tuple(types)

    @property
    def counts(self) -> Tuple[PositiveInt, ...]:
        """Returns number of elements in each field

        Returns:
            Tuple[PositiveInt, ...]: Tuple of number of elements in each field

        Note:
            If the field count is not 1, the count will be repeated for the number of its counts
            If you don't want this behavior, use the `pc.metadata.count` instead.

        >>> pc1.metadata.count
        (1, 1, 1)

        >>> pc2.metadata.count
        (1, 2, 1)

        >>> pc2.counts
        (1, 1, 1, 1)
        """

        return (1,) * sum(self.metadata.count)

    @property
    def points(self) -> int:
        """Returns number of points in the point cloud

        Returns:
            int: The number of points

        >>> pc.points
        1000
        """

        return self.metadata.points

    def numpy(self, fields: Optional[Sequence[str]] = None) -> npt.NDArray:
        """Returns numpy array of the point cloud

        Args:
            fields (Optional[Sequence[str]], optional): Fields to return.
                If None, all fields are returned. Defaults to None.

        Returns:
            npt.NDArray: Numpy array of the point cloud

        >>> pc.fields
        ("x", "y", "z")

        >>> pc.numpy()
        array([[0.0, 0.1, 0.2],
               [0.3, 0.4, 0.5],
               ...,
               [0.9, 1.0, 1.1]])

        >>> pc.numpy(("x", "y"))
        array([[0.0, 0.1],
               [0.3, 0.4],
               ...,
               [0.9, 1.0]])

        >>> pc.numpy(("x", "y", "z"))
        array([[0.0, 0.1, 0.2],
               [0.3, 0.4, 0.5],
               ...,
               [0.9, 1.0, 1.1]])
        """

        if fields is None:
            fields = self.fields

        if len(fields) == 0:
            return np.empty((0, 0))

        if self.metadata.points == 0:
            return np.empty((0, len(fields)))

        _stack = tuple(self.pc_data[field] for field in fields)

        return np.vstack(_stack).T

    def _save_as_ascii(self, fp: BinaryIO) -> None:
        """Saves point cloud to a file as a ascii

        Args:
            fp (BinaryIO): io buffer.
        """

        format = []
        for type, count in zip(self.metadata.type, self.metadata.count):
            if type == "F":
                format.extend(["%.10f"] * count)
            elif type == "I":
                format.extend(["%d"] * count)
            else:
                format.extend(["%u"] * count)

        np.savetxt(fp, self.pc_data, format)

    def _save_as_binary(self, fp: BinaryIO) -> None:
        """Saves point cloud to a file as a binary

        Args:
            path (PathLike): Path to the file
        """

        fp.write(self.pc_data.tobytes())

    def _save_as_binary_compressed(self, fp: BinaryIO) -> None:
        """Saves point cloud to a file as a binary compressed

        Args:
            fp (BinaryIO): io buffer.
        """

        uncompressed = b"".join(
            np.ascontiguousarray(self.pc_data[field]).tobytes()
            for field in self.pc_data.dtype.names  # type: ignore
        )

        if (compressed := lzf.compress(uncompressed)) is None:
            fp.seek(0)
            self.save(fp, encoding=Encoding.BINARY)
            return

        fp.write(struct.pack("II", len(compressed), len(uncompressed)))
        fp.write(compressed)

    def save(self, fp: Union[PathLike, BinaryIO], encoding: Encoding = Encoding.BINARY) -> None:
        """Saves point cloud to a file

        Args:
            fp (Union[PathLike, BinaryIO]): Path to the file or io buffer.
            encoding (Encoding, optional): Encoding to use. Defaults to Encoding.BINARY_COMPRESSED.
        """

        self.metadata.data = encoding

        is_open = False
        if isinstance(fp, Path):
            fp = fp.open(mode="wb")
            is_open = True
        elif isinstance(fp, str):
            fp = open(fp, mode="wb")
            is_open = True

        try:
            header = self.metadata.compose_header().encode()
            fp.write(header)

            if self.metadata.points < 1:
                return

            if encoding == Encoding.ASCII:
                self._save_as_ascii(fp)
            elif encoding == Encoding.BINARY:
                self._save_as_binary(fp)
            else:
                self._save_as_binary_compressed(fp)
        finally:
            if is_open:
                fp.close()

    def __add__(self, other: PointCloud) -> PointCloud:
        """Concatenates two point clouds together

        Args:
            other (PointCloud): Point cloud to concatenate with

        Returns:
            PointCloud: Concatenated point cloud

        >>> pc1 = PointCloud.from_xyz_points(np.random.rand(10000, 3))
        >>> pc2 = PointCloud.from_xyz_points(np.random.rand(10000, 3))

        >>> pc1 + pc2
        PointCloud(
            fields=('x', 'y', 'z') size=(4, 4, 4) type=('F', 'F', 'F') count=(1, 1, 1) points=20000
            width=20000 height=1 version='0.7' viewpoint=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            data=<Encoding.BINARY_COMPRESSED: 'binary_compressed'>
        )
        """

        if self.fields != other.fields:
            raise ValueError(
                "add: Can't concatenate point clouds with different fields. "
                f"({self.fields} vs. {other.fields})"
            )
        if self.types != other.types:
            raise ValueError(
                "add: Can't concatenate point clouds with different types. "
                f"({self.types} vs. {other.types})"
            )

        concatenated_pc = PointCloud.from_points(
            np.vstack((self.numpy(), other.numpy())), self.fields, self.types
        )

        return concatenated_pc

    def __getitem__(
        self, subscript: Union[slice, str, list[str], tuple[str, ...], npt.NDArray[np.bool_]]
    ) -> PointCloud:
        """Returns a point cloud with only the points that match the subscript

        Args:
            subscript (Union[slice, npt.NDArray[np.bool_]]): Subscript to match.

        Returns:
            PointCloud: Point cloud with only the points that match the subscript.

        >>> pc = PointCloud.from_xyz_points(np.random.rand(10000, 3))

        >>> pc[3:8]
        PointCloud(
            fields=('x', 'y', 'z') size=(4, 4, 4) type=('F', 'F', 'F') count=(1, 1, 1) points=5
            width=5 height=1 version='0.7' viewpoint=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            data=<Encoding.BINARY_COMPRESSED: 'binary_compressed'>
        )

        >>> mask = (pc.pc_data["x"] > 0.5) & (pc.pc_data["y"] < 0.5)
        >>> pc[mask]
        PointCloud(
            fields=('x', 'y', 'z') size=(4, 4, 4) type=('F', 'F', 'F') count=(1, 1, 1) points=2593
            width=2593 height=1 version='0.7' viewpoint=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            data=<Encoding.BINARY_COMPRESSED: 'binary_compressed'>
        )

        >>> pc[("x", "y")]
        PointCloud(
            fields=('x', 'y') size=(4, 4) type=('F', 'F') count=(1, 1) points=10000
            width=10000 height=1 version='0.7' viewpoint=(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            data=<Encoding.BINARY_COMPRESSED: 'binary_compressed'>
        )
        """

        fields = self.fields
        types = self.types
        if isinstance(subscript, slice):
            points_list = tuple(
                cast(npt.NDArray, self.pc_data[field][subscript])
                for field in self.pc_data.dtype.names  # type: ignore[assignment,union-attr]
            )
        elif isinstance(subscript, np.ndarray):
            mask = subscript.squeeze()
            if mask.ndim != 1:
                raise ValueError(f"Mask array must be 1-dimensional but got {mask.ndim}")

            points_list = tuple(
                cast(npt.NDArray, self.pc_data[field][mask])
                for field in self.pc_data.dtype.names  # type: ignore[assignment,union-attr]
            )
        elif isinstance(subscript, str) or all(isinstance(s, str) for s in cast(tuple, subscript)):
            if isinstance(subscript, str):
                subscript = (subscript,)

            subscript = cast(tuple, subscript)

            if not np.isin(subscript, self.fields).all():
                raise ValueError(f"Invalid field name(s): {subscript}")

            points_list = tuple(cast(npt.NDArray, self.pc_data[field]) for field in subscript)
            fields = tuple(subscript)
            types = tuple(self.pc_data[field].dtype for field in subscript)
        else:
            raise ValueError(f"Invalid subscript type: {type(subscript).__name__}")

        return PointCloud.from_points(points_list, fields, types)

    def __str__(self) -> str:
        return f"PointCloud({self.metadata})"

    def __len__(self) -> int:
        return self.points
