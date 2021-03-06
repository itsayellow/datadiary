# setup for datadiary

# import os.path
from setuptools import setup, find_packages

# here = os.path.abspath(os.path.dirname(__file__))

setup(
    name="datadiary",
    version="0.2",
    description="Generate html reports.",
    author="Matthew A. Clapp",
    author_email="itsayellow+dev@gmail.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    keywords="report",
    url="https://github.com/itsayellow/datadiary",
    include_package_data=True,  # so we get html files in datadiary/templates
    packages=["datadiary"],
    install_requires=[
        "Jinja2",
        "pandas",
        "numpy",
        "matplotlib",
        "tensorflow",
        "keras",
        "pydot",
        #'Pillow',
        "imagesize",
        "tqdm",
    ],
    entry_points={"console_scripts": ["diarygen=datadiary.command_line:main"]},
    python_requires=">=3",
)
