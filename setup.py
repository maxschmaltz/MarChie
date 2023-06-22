import setuptools

desc = '''
    MarChie: a Compact Open Source Tool for Analyzing Discrete Markov Chains.
    '''
with open('./README.md') as readme: long_description = readme.read()

setuptools.setup(
    name='marchie',
    version='0.3',
    author='Max Schmaltz',
    author_email='schmaltzmax@gmail.com',
    description=desc,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/maxschmaltz/MarChie',
    # packages=[
    #     'marchie'
    # ],
    packages = setuptools.find_packages(),
    install_requires=[
        'numpy',
        'graphviz'
    ],
    license='Apache Software License',
    keywords='markov chain, mathematics',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Education',
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ]
)