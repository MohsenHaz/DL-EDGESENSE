from setuptools import setup, find_packages

setup(
    name="dl_egsense",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=open("requirements.txt").read().splitlines(),
)
