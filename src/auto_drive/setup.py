from setuptools import setup
import os
from glob import glob

package_name = 'auto_drive'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 显式指定每个文件，避免 glob 问题
        (os.path.join('share', package_name, 'launch'), 
         ['launch/auto_drive_env.launch.py']),
        (os.path.join('share', package_name, 'world'), 
         glob('world/*.dae')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='abc',
    maintainer_email='abc@todo.todo',
    description='Auto Drive Package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
           'auto_drive_node = auto_drive.auto_drive_node:main',
           'effort_drive_node = auto_drive.effort_drive_node:main',
           'mesh_scan_node = auto_drive.mesh_scan_node:main',

        ],
    },
)

