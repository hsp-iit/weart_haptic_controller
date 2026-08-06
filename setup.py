from glob import glob
from setuptools import find_packages, setup

package_name = "gelsight_weart_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "weartsdk"],
    zip_safe=True,
    maintainer="your-name",
    maintainer_email="you@example.com",
    description="Bridge AI4CE GelSight tactile point clouds to a WEART haptic device.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "bridge_node = gelsight_weart_bridge.bridge_node:main",
        ],
    },
)
