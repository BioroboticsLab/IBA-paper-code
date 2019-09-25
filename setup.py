import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="attribution_bottleneck",
    version="0.0.1",
    author="Anonymous",
    author_email="",
    description="Implementation of the Attribution Bottleneck",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_namespace_packages(include=['attribution_bottleneck.*']),
    install_requires=[
        'numpy',
        'imageio',
        'torch',
        'torchvision',
    ],
    python_requires='>=3.6',
)
