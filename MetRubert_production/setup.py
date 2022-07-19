
from setuptools import setup, find_packages

# create setup
setup(
    name='MetRuBERT',
    version='0.1.0',
    description='A prouction model for metaphor prediction in dutch text',
    author='Joost Grunwald',
    author_email='joostgrunwald2001@gmail.com',
    packages=find_packages(include=['MetRuBERT', 'MetRuBERT.*']),
    install_requires=[
        'boto3>=1.16.63',
        'nltk==3.5',
        'numpy==1.20.0',
        'requests==2.25.1',
        'scikit-learn==0.24.1',
        'scipy==1.6.0',
        'torch==1.6.0',
        'torchvision==0.7.0',
        'tqdm==4.56.0',
        'transformers==4.2.2'
    ],
)
