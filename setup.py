from setuptools import setup, find_packages

setup(
    name='tensorwich',
    version='0.1.0',
    author='Kameron Brooks',
    author_email='kameron.cw.11@gmail.com',
    description='A library for saving embeddings for pandas dataframes',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/kameronbrooks/tensorwich',
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas"
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)