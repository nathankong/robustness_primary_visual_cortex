from setuptools import setup, find_packages

setup(
    name="robust_spectrum",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "torch==1.5.1",
        "joblib==0.16.0",
        "torchvision==0.6.1",
        "pandas==1.1.3",
        "eagerpy==0.29.0",
        "scipy==1.4.1",
        "foolbox==3.1.1",
        "numpy==1.19.0",
        "scikit-learn==0.23.1",
    ]
)
