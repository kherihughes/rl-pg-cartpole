from setuptools import setup, find_packages

setup(
    name="rl-pg-cartpole",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.9.0",
        "numpy>=1.19.2",
        "gymnasium>=0.29.0",
        "matplotlib>=3.3.2",
        "wandb>=0.12.0",
    ],
    author="Kheri Hughes",
    author_email="kherihughes@gmail.com",
    description="A PyTorch implementation of Policy Gradient with Value Function Baseline",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/kherihughes/rl-pg-cartpole",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 