# setup.py
from setuptools import setup, find_packages

setup(
    name="ai_playground",  # package name
    version="0.1.0",
    packages=find_packages(where="src"),  # looks inside src/ for packages
    package_dir={"": "src"},  # tells setuptools that packages are in src/
    include_package_data=True,  # include MANIFEST.in files if any
    install_requires=[
        "torch",
        "numpy",
        "pyyaml",
        "pydantic",
        "uvicorn",
    ],
    python_requires=">=3.14",
    entry_points={},
)
