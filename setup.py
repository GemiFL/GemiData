from setuptools import setup, find_packages

setup(name="gemi_data",
      version="0.0.1",
      packages=find_packages(),
      install_requires=["setuptools", "grpcio", "protobuf"],
      description="Gemi data generation")
