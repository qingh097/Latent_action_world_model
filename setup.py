from setuptools import find_packages, setup

setup(
    name="robot_wm",
    version="0.1.0",
    license="MIT",
    packages=find_packages(include=["robot_wm", "robot_wm.*"]),
)
