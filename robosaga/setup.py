from setuptools import find_packages, setup

setup(
    name="robosaga",
    version="0.1.0", 
    description="Saliency-guided Augmentation for Visuomotor Policy Learning",
    author="Zheyu Zhuang",
    author_email="zheyuzh@kth.se",
    packages=find_packages(),
    install_requires=["torch", "torchvision", "numpy", "robomimic"],
)
