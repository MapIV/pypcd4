from pathlib import Path

import pytest


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
def pcd_header_with_no_points():
    return [
        "VERSION .7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "WIDTH 0",
        "HEIGHT 0",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS 0",
        "DATA ascii",
    ]


@pytest.fixture
def pcd_header_with_multi_count_fields():
    return [
        "VERSION .7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 4",
        "WIDTH 213",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS 213",
        "DATA ascii",
    ]


@pytest.fixture
def pcd_header_with_negative_viewpoint():
    return [
        "VERSION .7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "WIDTH 213",
        "HEIGHT 1",
        "VIEWPOINT 1 -2 3 1 -1 2 -3",
        "POINTS 213",
        "DATA ascii",
    ]


@pytest.fixture
def pcd_header_with_underscore_in_fields():
    return [
        "VERSION .7",
        "FIELDS _ y _",
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
def pcd_header_with_extra_items():
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
        "COLUMN a b c",
        "EXTRA hope do nothing",
    ]


@pytest.fixture
def pcd_header_with_invalid_version():
    return [
        "VERSION 999",
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
def pcd_header_with_invalid_field_type():
    return [
        "VERSION .7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE U S A",
        "COUNT 1 1 1",
        "WIDTH 213",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS 213",
        "DATA ascii",
    ]


@pytest.fixture
def pcd_header_with_invalid_data_type():
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
        "DATA iicsa",
    ]


@pytest.fixture
def pcd_header_with_invalid_viewpoint():
    return [
        "VERSION .7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "WIDTH 213",
        "HEIGHT 1",
        "VIEWPOINT 0",
        "POINTS 213",
        "DATA ascii",
    ]


@pytest.fixture
def pcd_header_with_negative_points():
    return [
        "VERSION .7",
        "FIELDS x y z",
        "SIZE 4 4 4",
        "TYPE F F F",
        "COUNT 1 1 1",
        "WIDTH 0",
        "HEIGHT 0",
        "VIEWPOINT 0 0 0 1 0 0 0",
        "POINTS -123",
        "DATA ascii",
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
def xyzrgb_ascii_path():
    return f"{Path(__file__).resolve().parent}/pcd/ascii.pcd"


@pytest.fixture
def xyzrgb_ascii_with_underscore_path():
    return f"{Path(__file__).resolve().parent}/pcd/ascii_with_underscore.pcd"


@pytest.fixture
def xyzrgb_ascii_with_empty_points_path():
    return f"{Path(__file__).resolve().parent}/pcd/ascii_with_empty_points.pcd"


@pytest.fixture
def xyzrgb_binary_path():
    return f"{Path(__file__).resolve().parent}/pcd/binary.pcd"


@pytest.fixture
def xyzrgb_binary_compressed_path():
    return f"{Path(__file__).resolve().parent}/pcd/binary_compressed.pcd"
