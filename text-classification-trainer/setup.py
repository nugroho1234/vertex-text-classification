from setuptools import find_packages
from setuptools import setup
REQUIRED_PACKAGES = [
    'gcsfs==2023.9.2',
    'torch==2.0.0+cu118',
    'google-cloud-bigquery-storage==2.22.0',
    'google-cloud-bigquery==3.11.4',
    'transformers==4.34.0',
    'livelossplot==0.5.5',
    'pandas==2.0.3',
    'scikit-learn==1.3.1',
    'six==1.15.0'
]
setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(), # Automatically find packages within this directory or below.
    include_package_data=True, # if packages include any data files, those will be packed together.
    description='Text classification with Transformers and BERT'
)