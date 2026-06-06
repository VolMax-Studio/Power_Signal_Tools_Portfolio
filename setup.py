from setuptools import setup, find_packages

setup(
    name="power-signal-tools",
    version="0.1.0",
    author="Ivan Nestorov",
    author_email="volmax.core@gmail.com",
    description="Domain-grounded utilities for power and signal analysis.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
