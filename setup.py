from setuptools import setup

setup(
    name="xavier",
    version="0.1.0",
    py_modules=["xavier"],
    author="rakki194",
    author_email="acsipont@gmail.com",
    description="FP8 quantization and scaling utilities.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/rakki194/xavier",  # Replace with your project's URL
    license="MIT",  # Or whatever license you chose (I see a LICENSE.md)
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Or your chosen license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.18.0",
        "safetensors>=0.3.1",
        "torch>=2.0.0",
        "matplotlib>=3.3.0",
    ],
)
