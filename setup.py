from setuptools import setup, find_packages

setup(
    name='reactive-optical-flow',
    version='0.1.0',
    description='Reactive optical flow UAV navigation',
    packages=find_packages(exclude=['tests', 'flow_logs']),
    install_requires=[
        'opencv-python>=4.7,<4.8',
        'numpy>=1.24,<1.25',
        'airsim>=1.8,<1.9',
        'msgpack-rpc-python>=0.4,<0.5',
        'msgpack>=1.0,<1.0.7',
        'pandas>=1.5,<1.6',
        'plotly>=5.15,<5.20',
        'scipy>=1.10,<1.11',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'hybrid-nav=main:main',
            'airsim-streamer=slam_bridge.stream_airsim_image:main',
        ],
    },
)
