from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


@dataclass
class builtin_interfaces__msg__Time:
    sec: int
    nanosec: int

    __msgtype__: ClassVar[str] = "builtin_interfaces/msg/Time"


Time = builtin_interfaces__msg__Time


@dataclass
class std_msgs__msg__Header:
    stamp: builtin_interfaces__msg__Time
    frame_id: str

    __msgtype__: ClassVar[str] = "std_msgs/msg/Header"


Header = std_msgs__msg__Header


@dataclass
class sensor_msgs__msg__PointField:
    name: str
    offset: int
    datatype: int
    count: int

    INT8: ClassVar[int] = 1
    UINT8: ClassVar[int] = 2
    INT16: ClassVar[int] = 3
    UINT16: ClassVar[int] = 4
    INT32: ClassVar[int] = 5
    UINT32: ClassVar[int] = 6
    FLOAT32: ClassVar[int] = 7
    FLOAT64: ClassVar[int] = 8

    __msgtype__: ClassVar[str] = "sensor_msgs/msg/PointField"


PointField = sensor_msgs__msg__PointField


@dataclass
class sensor_msgs__msg__PointCloud2:
    header: std_msgs__msg__Header
    height: int
    width: int
    fields: list[sensor_msgs__msg__PointField]
    is_bigendian: bool
    point_step: int
    row_step: int
    data: np.ndarray[Any, np.dtype[np.uint8]]
    is_dense: bool

    __msgtype__: ClassVar[str] = "sensor_msgs/msg/PointCloud2"


PointCloud2 = sensor_msgs__msg__PointCloud2


TYPE_MAPPINGS = [
    (PointField.INT8, np.dtype("int8")),
    (PointField.UINT8, np.dtype("uint8")),
    (PointField.INT16, np.dtype("int16")),
    (PointField.UINT16, np.dtype("uint16")),
    (PointField.INT32, np.dtype("int32")),
    (PointField.UINT32, np.dtype("uint32")),
    (PointField.FLOAT32, np.dtype("float32")),
    (PointField.FLOAT64, np.dtype("float64")),
]

PFTYPE_TO_NPTYPE = dict(TYPE_MAPPINGS)
NPTYPE_TO_PFTYPE = dict((nptype, pftype) for pftype, nptype in TYPE_MAPPINGS)

PFTYPE_SIZES = {
    PointField.INT8: 1,
    PointField.UINT8: 1,
    PointField.INT16: 2,
    PointField.UINT16: 2,
    PointField.INT32: 4,
    PointField.UINT32: 4,
    PointField.FLOAT32: 4,
    PointField.FLOAT64: 8,
}


def build_dtype(msg: PointCloud2) -> list[tuple[str, object]]:
    offset = 0
    dtypes: list[tuple[str, object]] = []
    for field in msg.fields:
        while offset < field.offset:
            dtypes.append((f"__{offset}", np.dtype("uint8")))
            offset += 1

        dtypes.append((field.name, PFTYPE_TO_NPTYPE[field.datatype]))
        offset += PFTYPE_SIZES[field.datatype]

    while offset < msg.point_step:
        dtypes.append((f"__{offset}", np.dtype("uint8")))
        offset += 1

    return dtypes


def pointcloud2_to_array(msg: PointCloud2) -> np.ndarray:
    dtype = build_dtype(msg)

    array = np.frombuffer(msg.data, dtype)
    array = np.hstack([array[n][:, None] for n in array.dtype.names if not n.startswith("__")])

    return array
