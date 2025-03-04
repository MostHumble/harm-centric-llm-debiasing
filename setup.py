from setuptools import setup, find_packages

setup(
    name="multi-llm-debiasing-framework",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'transformers',
        'tqdm',
        'pyyaml',
    ],
    extras_require={
        'visualization': [
            'streamlit>=1.24.0',
        ],
    },
) 