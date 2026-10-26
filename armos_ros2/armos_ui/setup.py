from setuptools import find_packages, setup

package_name = "armos_ui"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="krishnapranav",
    maintainer_email="krishnapranav.221me227@nitk.edu.in",
    description="armos ROS 2 user interfaces: Tk control desk for the sim bridge",
    license="MIT",
    entry_points={
        "console_scripts": [
            "armos_gui = armos_ui.gui:main",
        ],
    },
)
