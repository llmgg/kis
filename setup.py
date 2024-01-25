import sys
import os
import argparse
from setuptools import setup, find_packages

ROOT = os.path.dirname(__file__)


def get_long_description():
    with open(os.path.join(ROOT, 'README.md'), encoding='utf-8') as f:
        markdown_txt = f.read()
        return markdown_txt


def get_version():
    for line in open(os.path.join(ROOT, "version")).readlines():
        words = line.strip().split()
        if len(words) == 0:
            continue
        if words[0] == "version":
            return words[-1]
    return "0.0.0"


def get_requirements(filename):
    with open(os.path.join(ROOT, filename)) as f:
        return [line.strip() for line in f]


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-r', '--requirement', help='Optionally specify a different requirements file.',
    required=False
)
args, unparsed_args = parser.parse_known_args()
sys.argv[1:] = unparsed_args

if args.requirement is None:
    install_requires = get_requirements("requirements.txt")
else:
    install_requires = get_requirements(args.requirement)

args = dict(
    name='kis',

    version=get_version(),

    description='Knowledge Integration System',
    long_description=get_long_description(),

    url='https://url.for.kis',

    author='wxw',
    author_email='****@foxmail.com',
    maintainer_email='*****@gmail.com',

    license='****',

    python_requires='>=3.9',

    packages=find_packages(exclude=("test", "*.test", "test.*", "unit_test")),

    setup_requires=['pytest-runner'],
    tests_require=['pytest', 'pytest-cov', 'pillow'],

    extras_require={
        'optional': ['tensorboard', 'matplotlib'],
    },

    install_requires=install_requires,

    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3 :: Only',

    ],

)

setup(**args)
