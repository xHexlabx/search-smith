# setup.py

from setuptools import setup, find_packages

def parse_requirements(filename='requirements.txt'):
    with open(filename, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='search-smith',
    version='0.1.0',
    author='Tee Hemjinda',
    author_email='tee.hemjinda.work@gmail.com',
    description='A smart search tool for PDFs using vector similarity search.',

    # find_packages() จะค้นหาทุกแพ็กเกจในโปรเจกต์อัตโนมัติ (ในที่นี้คือโฟลเดอร์ search-smith)
    packages=find_packages(),

    # ดึง library ที่จำเป็นจากไฟล์ requirements.txt
    install_requires=parse_requirements(),

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
