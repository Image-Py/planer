from setuptools import setup

descr = """A powerful light-weight inference framework for CNN.
The aim of planer is to provide efficient and adaptable inference environment for CNN model.
Also in order to enlarge the application scope,
we support ONNX format, which enables the converting of trained model within various DL frameworks.
"""

if __name__ == '__main__':
    setup(name='planer',
        version='0.26',
        url='https://github.com/Image-Py/planer',
        description='Powerful Light Artificial NEuRon',
        long_description=descr,
        author='Y.Dong, YXDragon',
        author_email='yxdragon@imagepy.org',
        license='BSD 3-clause',
        packages=['planer'],
        package_data={},
        install_requires=[
            'numpy'
        ],
    )
