from pathlib import Path

import pytest

from pypcd4 import PointCloud


@pytest.fixture
def pcd_header():
    return [
        "VERSION .7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "WIDTH 213",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS 213",
        "DATA ascii",
    ]


@pytest.fixture
def pcd_header_random_order():
    return [
        "VIEWPOINT 0 0 0 1 0 0 0",
        "TYPE F F F",
        "COUNT 1 1 1",
        "FIELDS x y z",
        "WIDTH 213",
        "VERSION .7",
        "HEIGHT 1",
        "DATA ascii",
        "POINTS 213",
        "SIZE 4 4 4",
    ]


@pytest.fixture
def pcd_header_with_typical_comment():
    return [
        "# .PCD v.7 - Point Cloud Data file format",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "TYPE F F F",
        "COUNT 1 1 1",
        "FIELDS x y z",
        "WIDTH 213",
        "VERSION .7",
        "HEIGHT 1",
        "DATA ascii",
        "POINTS 213",
        "SIZE 4 4 4",
    ]


@pytest.fixture
def pcd_header_with_random_comments():
    return [
        "# comment",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "### comment",
        "TYPE F F F",
        "COUNT 1 1 1",
        "# comment",
        "# comment",
        "# comment",
        "FIELDS x y z",
        "WIDTH 213",
        "# comment ###",
        "VERSION .7",
        "HEIGHT 1",
        "DATA ascii",
        "# comment ###",
        "POINTS 213",
        "SIZE 4 4 4",
        "# comment",
    ]


@pytest.fixture
def pcd_header_with_empty_lines():
    return [
        "",
        "VERSION .7",
        "FIELDS x y z",
        "",
        "",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "",
        "WIDTH 213",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS 213",
        "DATA ascii",
        "",
        "",
        "",
    ]


@pytest.fixture
def pcd_header_with_missing_items():
    return [
        "VERSION .7",
        "FIELDS x y z",
        # "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "WIDTH 213",
        # "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        # "POINTS 213",
        "DATA ascii",
    ]


@pytest.fixture
def pcd_xyzrgb_ascii():
    PATH = f"{Path(__file__).resolve().parent}/pcd/ascii.pcd"

    return PointCloud.from_path(PATH)


@pytest.fixture
def pcd_xyzrgb_binary():
    PATH = f"{Path(__file__).resolve().parent}/pcd/binary.pcd"

    return PointCloud.from_path(PATH)


@pytest.fixture
def pcd_xyzrgb_binary_compressed():
    PATH = f"{Path(__file__).resolve().parent}/pcd/binary_compressed.pcd"

    return PointCloud.from_path(PATH)
