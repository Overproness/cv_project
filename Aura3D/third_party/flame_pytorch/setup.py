from setuptools import setup, find_packages

setup(
    name="flame_pytorch",
    version="0.1.0",
    description="Lightweight PyTorch FLAME 3D face model wrapper",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=["torch", "numpy"],
)
