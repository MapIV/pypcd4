# pypcd4

## Description

pypcd4 is a modern reimagining of the original [pypcd](https://github.com/dimatura/pypcd) library,
offering enhanced capabilities and performance for working with Point Cloud Data (PCD) files.

This library builds upon the foundation laid by the original pypcd while incorporating modern
Python3 syntax and methodologies to provide a more efficient and user-friendly experience.

## Installation

To get started with `pypcd4`, install it using pip:

```shell
pip install pypcd4
```

## Usage

Let’s walk through some examples of how you can use pypcd4:

### Getting Started

First, import the PointCloud class from pypcd4:

```python
from pypcd4 import PointCloud
```

### Working with .pcd Files

If you have a .pcd file, you can read it into a PointCloud object:

```python
pc: PointCloud = PointCloud.from_path("point_cloud.pcd")

pc.fields
# ('x', 'y', 'z', 'intensity')
```

### Converting Between PointCloud and NumPy Array

You can convert a PointCloud to a NumPy array:

```python
array: np.ndarray = pc.numpy()

array.shape
# (1000, 4)
```

You can also specify the fields you want to include in the conversion:

```python
array: np.ndarray = pc.numpy(("x", "y", "z"))

array.shape
# (1000, 3)
```

And you can convert a NumPy array back to a PointCloud.
The method you use depends on the fields in your array:

```python
# If the array has x, y, z, and intensity fields,
pc = PointCloud.from_xyzi_points(array)

# Or if the array has x, y, z, and label fields,
pc = PointCloud.from_xyzl_points(array, label_type=np.uint32)
```

#### Creating Custom Conversion Methods

If you can’t find your preferred point type in the pre-defined conversion methods,
you can create your own:

```python
fields = ("x", "y", "z", "intensity", "new_field")
types = (np.float32, np.float32, np.float32, np.float32, np.float64)

pc = PointCloud.from_points(array, fields, types)
```

### Working with ROS PointCloud2 Messages

As of v0.4.0, you can convert a ROS PointCloud2 Message to a PointCloud:

```python
def callback(msg):
    pc = PointCloud.from_msg(msg)

    pc.fields
    # ("x", "y", "z", "intensity", "ring", "time")
```

### Concatenating Two PointClouds

The `pypcd4` supports concatenating two `PointCloud` objects together using the `+` operator.
This can be very useful when you want to merge two point clouds into one.

Here's how you can use it:

```python
pc1: PointCloud = PointCloud.from_path("xyzi1.pcd")
pc2: PointCloud = PointCloud.from_path("xyzi2.pcd")

# Concatenate two PointClouds
pc3: PointCloud = pc1 + pc2
```

Please note that the two PointCloud objects must have the same fields and types. If they don’t, a `ValueError` will be raised.

### Filtering a PointCloud

The `pypcd4` library provides a convenient way to filter a `PointCloud` using a mask.
Here's an example of how you can use it:

```python
# Create a random PointCloud
pc = PointCloud.from_xyz_points(np.random.rand(100, 3))

# Create a mask
mask = (pc.pc_data["x"] > 0.5) & (pc.pc_data["y"] < 0.5)

# Apply the mask to the PointCloud
filtered_pc = pc[mask]

# The filtered PointCloud only includes the points that match the mask
print(filtered_pc.numpy())
```

Please note that the mask must be a 1-dimensional array. If it’s not, a `ValueError` will be raised.

### Saving Your Work

Finally, you can save your PointCloud as a .pcd file:

```python
pc.save("nice_point_cloud.pcd")
```

## License

The library was rewritten and does not borrow any code from the original [pypcd](https://github.com/dimatura/pypcd) library.
Since it was heavily inspired by the original author's work, we extend his original BSD 3-Clause License and include his Copyright notice.
