from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
import numpy.typing as npt


@dataclass
class builtin_interfaces__msg__Time:
    sec: int
    nanosec: int

    __msgtype__: ClassVar[str] = "builtin_interfaces/msg/Time"


@dataclass
class std_msgs__msg__Header:
    stamp: builtin_interfaces__msg__Time
    frame_id: str

    __msgtype__: ClassVar[str] = "std_msgs/msg/Header"


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


TYPE_MAPPINGS: list[tuple[int, npt.DTypeLike]] = [
    (sensor_msgs__msg__PointField.INT8, np.dtype("int8")),
    (sensor_msgs__msg__PointField.UINT8, np.dtype("uint8")),
    (sensor_msgs__msg__PointField.INT16, np.dtype("int16")),
    (sensor_msgs__msg__PointField.UINT16, np.dtype("uint16")),
    (sensor_msgs__msg__PointField.INT32, np.dtype("int32")),
    (sensor_msgs__msg__PointField.UINT32, np.dtype("uint32")),
    (sensor_msgs__msg__PointField.FLOAT32, np.dtype("float32")),
    (sensor_msgs__msg__PointField.FLOAT64, np.dtype("float64")),
]

PFTYPE_TO_NPTYPE = dict(TYPE_MAPPINGS)
NPTYPE_TO_PFTYPE = dict((nptype, pftype) for pftype, nptype in TYPE_MAPPINGS)

PFTYPE_SIZES = {
    sensor_msgs__msg__PointField.INT8: 1,
    sensor_msgs__msg__PointField.UINT8: 1,
    sensor_msgs__msg__PointField.INT16: 2,
    sensor_msgs__msg__PointField.UINT16: 2,
    sensor_msgs__msg__PointField.INT32: 4,
    sensor_msgs__msg__PointField.UINT32: 4,
    sensor_msgs__msg__PointField.FLOAT32: 4,
    sensor_msgs__msg__PointField.FLOAT64: 8,
}


def build_dtype_from_msg(msg: sensor_msgs__msg__PointCloud2) -> list[tuple[str, npt.DTypeLike]]:
    offset = 0
    dtypes: list[tuple[str, npt.DTypeLike]] = []
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
