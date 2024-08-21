from setuptools import setup, find_packages

setup(
    name='liver_segmentation_model',
    version='1.0',
    description='A liver segmentation model using U-Net architecture with MONAI framework.',
    author='satya srinivas',
    url='https://github.com/SatyaSrinivas12/Liver_segmentation.git',
    packages=find_packages(),  
    install_requires=[
        'torch',
        'monai',
        'numpy',
        'matplotlib',
        'tqdm',
        'nibabel' ,
    ],
    python_requires='>=3.10.0'
)
