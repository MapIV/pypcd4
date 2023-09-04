from typing import List

import numpy as np
import pytest
from pydantic import ValidationError

from pypcd4 import MetaData, PointCloud


def test_parse_header(pcd_header: List[str]):
    header = MetaData.parse_header(pcd_header)

    assert header.version == ".7"
    assert header.fields == ("x", "y", "z")
    assert header.size == (4, 4, 4)
    assert header.type == ("F", "F", "F")
    assert header.count == (1, 1, 1)
    assert header.width == 213
    assert header.height == 1
    assert header.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert header.points == 213
    assert header.data == "ascii"


def test_parse_header_random_order(pcd_header_random_order: List[str]):
    header = MetaData.parse_header(pcd_header_random_order)

    assert header.version == ".7"
    assert header.fields == ("x", "y", "z")
    assert header.size == (4, 4, 4)
    assert header.type == ("F", "F", "F")
    assert header.count == (1, 1, 1)
    assert header.width == 213
    assert header.height == 1
    assert header.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert header.points == 213
    assert header.data == "ascii"


def test_parse_header_with_typical_comments(pcd_header_with_typical_comment: List[str]):
    header = MetaData.parse_header(pcd_header_with_typical_comment)

    assert header.version == ".7"
    assert header.fields == ("x", "y", "z")
    assert header.size == (4, 4, 4)
    assert header.type == ("F", "F", "F")
    assert header.count == (1, 1, 1)
    assert header.width == 213
    assert header.height == 1
    assert header.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert header.points == 213
    assert header.data == "ascii"


def test_parse_header_with_random_comments(pcd_header_with_random_comments: List[str]):
    header = MetaData.parse_header(pcd_header_with_random_comments)

    assert header.version == ".7"
    assert header.fields == ("x", "y", "z")
    assert header.size == (4, 4, 4)
    assert header.type == ("F", "F", "F")
    assert header.count == (1, 1, 1)
    assert header.width == 213
    assert header.height == 1
    assert header.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert header.points == 213
    assert header.data == "ascii"


def test_parse_header_with_empty_lines(pcd_header_with_empty_lines: List[str]):
    header = MetaData.parse_header(pcd_header_with_empty_lines)

    assert header.version == ".7"
    assert header.fields == ("x", "y", "z")
    assert header.size == (4, 4, 4)
    assert header.type == ("F", "F", "F")
    assert header.count == (1, 1, 1)
    assert header.width == 213
    assert header.height == 1
    assert header.viewpoint == (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert header.points == 213
    assert header.data == "ascii"


def test_parse_header_with_missing_items(pcd_header_with_missing_items: List[str]):
    with pytest.raises(ValidationError):
        MetaData.parse_header(pcd_header_with_missing_items)


def test_load_ascii_pcd(pcd_xyzrgb_ascii: PointCloud):
    assert pcd_xyzrgb_ascii.metadata.data == "ascii"
    assert pcd_xyzrgb_ascii.pc_data.dtype.names == pcd_xyzrgb_ascii.metadata.fields
    assert len(pcd_xyzrgb_ascii.pc_data) == pcd_xyzrgb_ascii.metadata.points


def test_load_binary_pcd(pcd_xyzrgb_binary: PointCloud):
    assert pcd_xyzrgb_binary.metadata.data == "binary"
    assert pcd_xyzrgb_binary.pc_data.dtype.names == pcd_xyzrgb_binary.metadata.fields
    assert len(pcd_xyzrgb_binary.pc_data) == pcd_xyzrgb_binary.metadata.points


def test_load_binary_compressed_pcd(pcd_xyzrgb_binary_compressed: PointCloud):
    assert pcd_xyzrgb_binary_compressed.metadata.data == "binary_compressed"
    assert (
        pcd_xyzrgb_binary_compressed.pc_data.dtype.names
        == pcd_xyzrgb_binary_compressed.metadata.fields
    )
    assert len(pcd_xyzrgb_binary_compressed.pc_data) == pcd_xyzrgb_binary_compressed.metadata.points


def test_from_points():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    fields = ("x", "y", "z")
    types = (np.float32, np.float32, np.float32)

    pc = PointCloud.from_points(array, fields, types)

    assert pc.fields == fields
    assert pc.types == types
    assert pc.points == 2
    assert pc.pc_data.dtype.names == fields

    assert np.array_equal(pc.pc_data["x"], array.T[0])
    assert np.array_equal(pc.pc_data["y"], array.T[1])
    assert np.array_equal(pc.pc_data["z"], array.T[2])

    assert pc.pc_data["x"].dtype == np.dtype(types[0])
    assert pc.pc_data["y"].dtype == np.dtype(types[1])
    assert pc.pc_data["z"].dtype == np.dtype(types[2])
