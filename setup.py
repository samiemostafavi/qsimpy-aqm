import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()
setup(
  name="qsimpy-aqm",
  version="0.0.1",
  description="Active queue management and queueing simulation using SimPy",
  long_description=README,
  long_description_content_type="text/markdown",
  author="Seyed Samie Mostafavi",
  author_email="samiemostafavi@gmail.com",
  license="MIT",
  packages=find_packages(include=['qsimpy_aqm'], exclude=['arrivals','test','utils']),
  zip_safe=False
)