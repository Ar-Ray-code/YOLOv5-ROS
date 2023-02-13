from setuptools import setup

import os
from glob import glob
from urllib.request import urlretrieve
from setuptools import find_packages

package_name = 'yolov5_ros'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('./launch/*.launch.py')),
        # (os.path.join('share', package_name), glob('../weights/*.pth'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Ar-Ray-code',
    author_email="ray255ar@gmail.com",
    maintainer='Ar-Ray-code',
    maintainer_email="ray255ar@gmail.com",
    description='YOLOv5 + ROS2 Foxy',
    license='GNU GENERAL PUBLIC LICENSE Version 3',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov5_ros = '+package_name+'.main:ros_main',
        ],
    },
)
