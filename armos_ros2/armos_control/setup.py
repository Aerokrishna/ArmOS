from glob import glob

from setuptools import find_packages, setup

package_name = "armos_control"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/config/peg_hole", glob("config/peg_hole/*.yaml") + glob("config/peg_hole/*.xml")),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="krishnapranav",
    maintainer_email="krishnapranav.221me227@nitk.edu.in",
    description="armos ROS 2 nodes: MuJoCo sim bridge (robot-compatible topics) and the admittance controller "
                "(sim and real)",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "armos_ros_sim = armos_control.sim_bridge:main",
            "admittance_controller = armos_control.admittance:main",
        ],
    },
)
