from setuptools import setup, find_packages

setup(
    name="FTIV",
    version="0.00",
    packages=find_packages(),  # Automatically finds sub-packages
    install_requires=[
        "rasterio",
        "pyproj",
        "scipy",
        "matplotlib",
        "tqdm"

    ],  # Add other necessary dependencies
    python_requires=">=3.7",  # Specify the Python version requirement
    entry_points={
        'console_scripts': [
            'ftiv=ftiv.main_compact:main',
        ],
    }
)