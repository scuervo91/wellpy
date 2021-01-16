import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="wellpy", # Replace with your own username
    version="v0.1",
    author="Santiago Cuervo",
    author_email="scuervo91@gmail.com",
    description="Library that contains the Well and WellsGroup Classes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scuervo91/wellpy",
    download_url="https://github.com/scuervo91/wellpy/archive/v0.1.tar.gz",
    packages=setuptools.find_packages(),
    include_package_data = True,
    package_data = {'':['*.csv','*.json']},
    install_requires=[            
        'numpy',
        'pandas',
        'scipy',
        'geopandas',
        'pyvista',
        'shapely',
        'folium',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
