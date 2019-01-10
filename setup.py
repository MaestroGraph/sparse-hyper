from setuptools import setup

setup(name='sparse',
      version='0.1',
      description='Sparse layer',
      url='http://github.com/MaestroGraph/sparse-hyper',
      author='Peter Bloem',
      author_email='sparse@peterbloem.nlß',
      license='MIT',
      packages=['sparse'],
      install_requires=[
            'matplotlib',
            'torch',
            'tqdm'
      ],
      zip_safe=False)