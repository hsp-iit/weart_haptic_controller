from setuptools import find_packages, setup


package_name = "weart_combined_controller"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "weartsdk"],
    zip_safe=True,
    maintainer="your-name",
    maintainer_email="you@example.com",
    description="Single-client WEART raw tracking publisher and GelSight haptic bridge.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "combined_controller = weart_combined_controller.combined_controller:main",
        ],
    },
)
