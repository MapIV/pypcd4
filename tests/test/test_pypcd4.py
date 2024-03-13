from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from pydantic import ValidationError

from pypcd4 import Encoding, MetaData, PointCloud


def test_parse_pcd_header(pcd_header):
    metadata = MetaData.parse_header(pcd_header)

    assert metadata.version == ".7"
    assert metadata.fields == ("x", "y", "z")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 1)
    assert metadata.width == 213
    assert metadata.height == 1
    assert metadata.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert metadata.points == 213
    assert metadata.data == "ascii"


def test_parse_pcd_header_random_order(pcd_header_random_order):
    metadata = MetaData.parse_header(pcd_header_random_order)

    assert metadata.version == ".7"
    assert metadata.fields == ("x", "y", "z")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 1)
    assert metadata.width == 213
    assert metadata.height == 1
    assert metadata.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert metadata.points == 213
    assert metadata.data == "ascii"


def test_parse_pcd_header_with_typical_comments(pcd_header_with_typical_comment):
    metadata = MetaData.parse_header(pcd_header_with_typical_comment)

    assert metadata.version == ".7"
    assert metadata.fields == ("x", "y", "z")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 1)
    assert metadata.width == 213
    assert metadata.height == 1
    assert metadata.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert metadata.points == 213
    assert metadata.data == "ascii"


def test_parse_pcd_header_with_random_comments(pcd_header_with_random_comments):
    metadata = MetaData.parse_header(pcd_header_with_random_comments)

    assert metadata.version == ".7"
    assert metadata.fields == ("x", "y", "z")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 1)
    assert metadata.width == 213
    assert metadata.height == 1
    assert metadata.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert metadata.points == 213
    assert metadata.data == "ascii"


def test_parse_pcd_header_with_empty_lines(pcd_header_with_empty_lines):
    metadata = MetaData.parse_header(pcd_header_with_empty_lines)

    assert metadata.version == ".7"
    assert metadata.fields == ("x", "y", "z")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 1)
    assert metadata.width == 213
    assert metadata.height == 1
    assert metadata.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert metadata.points == 213
    assert metadata.data == "ascii"


def test_parse_pcd_header_with_no_points(pcd_header_with_no_points):
    metadata = MetaData.parse_header(pcd_header_with_no_points)

    assert metadata.version == ".7"
    assert metadata.fields == ("x", "y", "z")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 1)
    assert metadata.width == 0
    assert metadata.height == 0
    assert metadata.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert metadata.points == 0
    assert metadata.data == "ascii"


def test_parse_pcd_header_with_multi_count_fields(pcd_header_with_multi_count_fields):
    metadata = MetaData.parse_header(pcd_header_with_multi_count_fields)

    assert metadata.version == ".7"
    assert metadata.fields == ("x", "y", "z")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 4)
    assert metadata.width == 213
    assert metadata.height == 1
    assert metadata.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert metadata.points == 213
    assert metadata.data == "ascii"


def test_parse_pcd_header_with_negative_viewpoint(pcd_header_with_negative_viewpoint):
    metadata = MetaData.parse_header(pcd_header_with_negative_viewpoint)

    assert metadata.version == ".7"
    assert metadata.fields == ("x", "y", "z")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 1)
    assert metadata.width == 213
    assert metadata.height == 1
    assert metadata.viewpoint == (1.0, -2.0, 3.0, 1.0, -1.0, 2.0, -3.0)
    assert metadata.points == 213
    assert metadata.data == "ascii"


def test_parse_pcd_header_with_underscore_in_fields(pcd_header_with_underscore_in_fields):
    metadata = MetaData.parse_header(pcd_header_with_underscore_in_fields)

    assert metadata.version == ".7"
    assert metadata.fields != ("_", "y", "_")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 1)
    assert metadata.width == 213
    assert metadata.height == 1
    assert metadata.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert metadata.points == 213
    assert metadata.data == "ascii"


def test_parse_pcd_header_with_extra_items(pcd_header_with_extra_items):
    metadata = MetaData.parse_header(pcd_header_with_extra_items)

    assert metadata.version == ".7"
    assert metadata.fields == ("x", "y", "z")
    assert metadata.size == (4, 4, 4)
    assert metadata.type == ("F", "F", "F")
    assert metadata.count == (1, 1, 1)
    assert metadata.width == 213
    assert metadata.height == 1
    assert metadata.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert metadata.points == 213
    assert metadata.data == "ascii"


def test_compose_pcd_header(pcd_header):
    metadata = MetaData.parse_header(pcd_header)

    assert metadata.compose_header() == (
        "VERSION .7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        "WIDTH 213\nHEIGHT 1\nVIEWPOINT 0.0 0.0 0.0 1.0 0.0 0.0 0.0\nPOINTS 213\nDATA ascii\n"
    )


def test_build_dtype(pcd_header):
    metadata = MetaData.parse_header(pcd_header)

    assert metadata.build_dtype() == [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]


def test_build_dtype_with_multi_count_fields(pcd_header_with_multi_count_fields):
    metadata = MetaData.parse_header(pcd_header_with_multi_count_fields)

    assert metadata.build_dtype() == [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z_0000", "<f4"),
        ("z_0001", "<f4"),
        ("z_0002", "<f4"),
        ("z_0003", "<f4"),
    ]


def test_parse_pcd_header_with_invalid_version(pcd_header_with_invalid_version):
    with pytest.raises(ValidationError):
        MetaData.parse_header(pcd_header_with_invalid_version)


def test_parse_pcd_header_with_invalid_field_type(pcd_header_with_invalid_field_type):
    with pytest.raises(ValidationError):
        MetaData.parse_header(pcd_header_with_invalid_field_type)


def test_parse_pcd_header_with_invalid_data_type(pcd_header_with_invalid_data_type):
    with pytest.raises(ValidationError):
        MetaData.parse_header(pcd_header_with_invalid_data_type)


def test_parse_pcd_header_with_invalid_viewpoint(pcd_header_with_invalid_viewpoint):
    with pytest.raises(ValidationError):
        MetaData.parse_header(pcd_header_with_invalid_viewpoint)


def test_parse_pcd_header_with_negative_points(pcd_header_with_negative_points):
    with pytest.raises(ValidationError):
        MetaData.parse_header(pcd_header_with_negative_points)


def test_parse_pcd_header_with_missing_items(pcd_header_with_missing_items):
    with pytest.raises(ValidationError):
        MetaData.parse_header(pcd_header_with_missing_items)


def test_load_ascii_pcd(xyzrgb_ascii_path):
    pc = PointCloud.from_path(xyzrgb_ascii_path)

    assert pc.metadata.data == "ascii"
    assert pc.pc_data.dtype.names == pc.metadata.fields
    assert len(pc.pc_data) == pc.metadata.points


def test_load_xyzrgb_ascii_with_underscore_pcd(xyzrgb_ascii_with_underscore_path):
    pc = PointCloud.from_path(xyzrgb_ascii_with_underscore_path)

    assert pc.metadata.data == "ascii"
    assert pc.pc_data.dtype.names == pc.metadata.fields
    assert len(pc.pc_data) == pc.metadata.points


def test_load_xyzrgb_ascii_with_empty_points_pcd(xyzrgb_ascii_with_empty_points_path):
    pc = PointCloud.from_path(xyzrgb_ascii_with_empty_points_path)

    assert pc.metadata.data == "ascii"
    assert pc.pc_data.dtype.names == pc.metadata.fields
    assert len(pc.pc_data) == pc.metadata.points


def test_load_binary_pcd(xyzrgb_binary_path):
    pc = PointCloud.from_path(xyzrgb_binary_path)

    assert pc.metadata.data == "binary"
    assert pc.pc_data.dtype.names == pc.metadata.fields
    assert len(pc.pc_data) == pc.metadata.points


def test_load_binary_compressed_pcd(xyzrgb_binary_compressed_path):
    pc = PointCloud.from_path(xyzrgb_binary_compressed_path)

    assert pc.metadata.data == "binary_compressed"
    assert pc.pc_data.dtype.names == pc.metadata.fields
    assert len(pc.pc_data) == pc.metadata.points


def test_from_points():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    fields = ("x", "y", "z")
    types = (np.float32, np.float32, np.float32)
    count = (1, 1, 1)

    pc = PointCloud.from_points(array, fields, types, count)

    assert pc.fields == fields
    assert pc.types == types
    assert pc.points == 2
    assert pc.count == count
    assert pc.pc_data.dtype.names == fields

    assert np.array_equal(pc.pc_data["x"], array.T[0])
    assert np.array_equal(pc.pc_data["y"], array.T[1])
    assert np.array_equal(pc.pc_data["z"], array.T[2])

    assert pc.pc_data["x"].dtype == np.dtype(types[0])
    assert pc.pc_data["y"].dtype == np.dtype(types[1])
    assert pc.pc_data["z"].dtype == np.dtype(types[2])


def test_from_points_with_invalid_points():
    array = "invalid_points"
    fields = ("x", "y", "z")
    types = (np.float32, np.float32, np.float32)

    with pytest.raises(TypeError):
        PointCloud.from_points(array, fields, types)


def test_from_points_with_invalid_fields_and_types():
    points = np.random.random((100, 4))
    fields = ("x", "y", "z", "a")
    types = (np.float32, np.float32)

    with pytest.raises(ValueError):
        PointCloud.from_points(points, fields, types)


def test_from_xyz_points():
    points = np.random.random((100, 3))

    pc = PointCloud.from_xyz_points(points)

    assert pc.fields == ("x", "y", "z")
    assert pc.types == (np.float32, np.float32, np.float32)


def test_from_xyzi_points():
    points = np.random.random((100, 4))

    pc = PointCloud.from_xyzi_points(points)

    assert pc.fields == ("x", "y", "z", "intensity")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32)


def test_from_xyzl_points():
    points = np.random.random((100, 4))

    label_type = np.int32
    pc = PointCloud.from_xyzl_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "label")
    assert pc.types == (np.float32, np.float32, np.float32, label_type)

    label_type = np.float32
    pc = PointCloud.from_xyzl_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "label")
    assert pc.types == (np.float32, np.float32, np.float32, label_type)

    label_type = np.uint8
    pc = PointCloud.from_xyzl_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "label")
    assert pc.types == (np.float32, np.float32, np.float32, label_type)


def test_from_xyzrgb_points():
    points = np.random.random((100, 4))

    pc = PointCloud.from_xyzrgb_points(points)

    assert pc.fields == ("x", "y", "z", "rgb")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32)


def test_from_xyzrgbl_points():
    points = np.random.random((100, 5))

    label_type = np.int32
    pc = PointCloud.from_xyzrgbl_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "rgb", "label")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, label_type)

    label_type = np.float32
    pc = PointCloud.from_xyzrgbl_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "rgb", "label")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, label_type)

    label_type = np.uint8
    pc = PointCloud.from_xyzrgbl_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "rgb", "label")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, label_type)


def test_from_xyzil_points():
    points = np.random.random((100, 5))

    label_type = np.int32
    pc = PointCloud.from_xyzil_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "intensity", "label")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, label_type)

    label_type = np.float32
    pc = PointCloud.from_xyzil_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "intensity", "label")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, label_type)

    label_type = np.uint8
    pc = PointCloud.from_xyzil_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "intensity", "label")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, label_type)


def test_from_xyzirgb_points():
    points = np.random.random((100, 5))

    pc = PointCloud.from_xyzirgb_points(points)

    assert pc.fields == ("x", "y", "z", "intensity", "rgb")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.float32)


def test_from_xyzirgbl_points():
    points = np.random.random((100, 6))

    label_type = np.int32
    pc = PointCloud.from_xyzirgbl_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "intensity", "rgb", "label")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.float32, label_type)

    label_type = np.float32
    pc = PointCloud.from_xyzirgbl_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "intensity", "rgb", "label")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.float32, label_type)

    label_type = np.uint8
    pc = PointCloud.from_xyzirgbl_points(points, label_type)

    assert pc.fields == ("x", "y", "z", "intensity", "rgb", "label")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.float32, label_type)


def test_from_xyzt_points():
    points = np.random.random((100, 5))

    pc = PointCloud.from_xyzt_points(points)

    assert pc.fields == ("x", "y", "z", "sec", "nsec")
    assert pc.types == (np.float32, np.float32, np.float32, np.uint32, np.uint32)


def test_from_xyzir_points():
    points = np.random.random((100, 5))

    pc = PointCloud.from_xyzir_points(points)

    assert pc.fields == ("x", "y", "z", "intensity", "ring")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.uint16)


def test_from_xyzirt_points():
    points = np.random.random((100, 6))

    pc = PointCloud.from_xyzirt_points(points)

    assert pc.fields == ("x", "y", "z", "intensity", "ring", "time")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.uint16, np.float32)


def test_from_xyzit_points():
    points = np.random.random((100, 5))

    pc = PointCloud.from_xyzit_points(points)

    assert pc.fields == ("x", "y", "z", "intensity", "timestamp")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.float64)


def test_from_xyzis_points():
    points = np.random.random((100, 5))

    pc = PointCloud.from_xyzis_points(points)

    assert pc.fields == ("x", "y", "z", "intensity", "stamp")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.float64)


def test_from_xyzisc_points():
    points = np.random.random((100, 6))

    pc = PointCloud.from_xyzisc_points(points)

    assert pc.fields == ("x", "y", "z", "intensity", "stamp", "classification")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.float64, np.uint8)


def test_from_xyzrgbs_points():
    points = np.random.random((100, 5))

    pc = PointCloud.from_xyzrgbs_points(points)

    assert pc.fields == ("x", "y", "z", "rgb", "stamp")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.float64)


def test_from_xyzirgbs_points():
    points = np.random.random((100, 6))

    pc = PointCloud.from_xyzirgbs_points(points)

    assert pc.fields == ("x", "y", "z", "intensity", "rgb", "stamp")
    assert pc.types == (np.float32, np.float32, np.float32, np.float32, np.float32, np.float64)


def test_from_xyzirgbsc_points():
    points = np.random.random((100, 7))

    pc = PointCloud.from_xyzirgbsc_points(points)

    assert pc.fields == ("x", "y", "z", "intensity", "rgb", "stamp", "classification")
    assert pc.types == (
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float64,
        np.uint8,
    )


def test_from_xyziradt_points():
    points = np.random.random((100, 9))

    pc = PointCloud.from_xyziradt_points(points)

    assert pc.fields == (
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
    assert pc.types == (
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


def test_from_ouster_points():
    points = np.random.random((100, 9))

    pc = PointCloud.from_ouster_points(points)

    assert pc.fields == (
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
    assert pc.types == (
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


def test_from_invalid_points():
    points = np.random.random((100, 1))

    with pytest.raises(ValueError):
        PointCloud.from_ouster_points(points)


def test_encode_rgb():
    r = np.array([249, 249, 249])
    g = np.array([230, 230, 230])
    b = np.array([206, 206, 206])

    expect = np.array((r << 16) | (g << 8) | (b << 0), np.uint32)
    expect.dtype = np.float32

    output = PointCloud.encode_rgb((r, g, b))

    assert np.allclose(output, expect)


def test_decode_rgb():
    r = np.array([249, 249, 249])
    g = np.array([230, 230, 230])
    b = np.array([206, 206, 206])

    input = np.array((r << 16) | (g << 8) | (b << 0), np.uint32)
    input.dtype = np.float32

    expect = np.array([r, g, b], dtype=np.uint8)
    output = PointCloud.decode_rgb(input)

    assert np.allclose(output, expect.T)


def test_numpy():
    in_points = np.random.random((100, 3))
    fields = ("x", "y", "z")
    types = (np.float32, np.float32, np.float32)
    pc = PointCloud.from_points(in_points, fields, types)

    out_points = pc.numpy()
    assert np.allclose(out_points, in_points)

    out_points = pc.numpy(fields=("x", "y"))
    assert np.allclose(out_points, in_points[:, :2])

    out_points = pc.numpy(fields=[])
    assert np.allclose(out_points, np.empty((0, 0)))


def test_save_as_ascii():
    in_points = np.random.randint(0, 1000, (100, 3))
    fields = ("x", "y", "z")
    types = (np.float32, np.int8, np.uint64)
    pc = PointCloud.from_points(in_points, fields, types)

    with TemporaryDirectory() as dirname:
        path = dirname / Path("test.pcd")

        pc.save(path, Encoding.ASCII)
        assert path.is_file()

        pc2 = PointCloud.from_path(path)
        assert np.allclose(pc2.numpy(), pc.numpy())
        assert pc2.fields == pc.fields
        assert pc2.types == pc.types
        assert pc2.points == pc.points
        assert pc2.metadata.data == pc.metadata.data


def test_save_as_binary():
    in_points = np.random.randint(0, 1000, (100, 3))
    fields = ("x", "y", "z")
    types = (np.float32, np.int8, np.uint64)
    pc = PointCloud.from_points(in_points, fields, types)

    with TemporaryDirectory() as dirname:
        path = dirname / Path("test.pcd")

        pc.save(path, Encoding.BINARY)
        assert path.is_file()

        pc2 = PointCloud.from_path(path)
        assert np.allclose(pc2.numpy(), pc.numpy())
        assert pc2.fields == pc.fields
        assert pc2.types == pc.types
        assert pc2.points == pc.points
        assert pc2.metadata.data == pc.metadata.data


def test_save_as_binary_compressed():
    in_points = np.random.randint(0, 1000, (100, 3))
    fields = ("x", "y", "z")
    types = (np.float32, np.int8, np.uint64)
    pc = PointCloud.from_points(in_points, fields, types)

    with TemporaryDirectory() as dirname:
        path = dirname / Path("test.pcd")

        pc.save(path, Encoding.BINARY_COMPRESSED)
        assert path.is_file()

        pc2 = PointCloud.from_path(path)
        assert np.allclose(pc2.numpy(), pc.numpy())
        assert pc2.fields == pc.fields
        assert pc2.types == pc.types
        assert pc2.points == pc.points
        assert pc2.metadata.data == pc.metadata.data


def test_save_with_empty_points():
    in_points = np.empty((0, 3))
    fields = ("x", "y", "z")
    types = (np.float32, np.float32, np.float32)
    pc = PointCloud.from_points(in_points, fields, types)

    with TemporaryDirectory() as dirname:
        path = dirname / Path("test.pcd")

        pc.save(path)
        assert path.is_file()

        pc2 = PointCloud.from_path(path)
        assert np.allclose(pc2.numpy(), pc.numpy())
        assert pc2.fields == pc.fields
        assert pc2.types == pc.types
        assert pc2.points == pc.points


def test_save_to_file_buffer():
    in_points = np.random.randint(0, 1000, (100, 3))
    fields = ("x", "y", "z")
    types = (np.float32, np.int8, np.uint64)
    pc = PointCloud.from_points(in_points, fields, types)

    with TemporaryDirectory() as dirname:
        path = dirname / Path("test.pcd")

        with path.open("wb") as fp:
            pc.save(fp, Encoding.BINARY)
        assert path.is_file()

        pc2 = PointCloud.from_path(path)
        assert np.allclose(pc2.numpy(), pc.numpy())
        assert pc2.fields == pc.fields
        assert pc2.types == pc.types
        assert pc2.points == pc.points
        assert pc2.metadata.data == pc.metadata.data


def test_save_to_bytes_io():
    in_points = np.random.randint(0, 1000, (100, 3))
    fields = ("x", "y", "z")
    types = (np.float32, np.int8, np.uint64)
    pc = PointCloud.from_points(in_points, fields, types)

    buffer = BytesIO()
    pc.save(buffer, Encoding.BINARY)
    buffer.seek(0)

    pc2 = PointCloud.from_fileobj(buffer)
    assert np.allclose(pc2.numpy(), pc.numpy())
    assert pc2.fields == pc.fields
    assert pc2.types == pc.types
    assert pc2.points == pc.points
    assert pc2.metadata.data == pc.metadata.data


def test_pointcloud_concatenation():
    in_points = np.random.randint(0, 1000, (100, 3))
    fields = ("x", "y", "z")
    types = (np.float32, np.int8, np.uint64)
    pc = PointCloud.from_points(in_points, fields, types)

    # Can concatenate PointCloud
    in_points = np.random.randint(0, 1000, (100, 3))
    pc2 = PointCloud.from_points(in_points, fields, types)
    pc3 = pc + pc2
    assert pc3.points == 200
    assert len(pc3.pc_data) == 200
    assert len(pc3.pc_data.dtype) == 3
    assert pc3.fields == pc.fields
    assert pc3.types == pc.types
    assert pc3.metadata.data == pc.metadata.data
    assert np.allclose(pc3.numpy(), np.concatenate((pc.numpy(), pc2.numpy()), axis=0))

    # Cannot concatenate fields are different
    pc4 = PointCloud.from_points(in_points, ("x", "y", "zz"), types)
    with pytest.raises(ValueError):
        pc + pc4

    # Cannot concatenate because types are different
    pc5 = PointCloud.from_points(in_points, fields, (np.float32, np.int8, np.float32))
    with pytest.raises(ValueError):
        pc + pc5
