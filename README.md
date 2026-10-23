# Description
This module reads a PETSIRD data file and generates a 3D mesh file.

## How to install

You need PETSIRD and trimesh. If generating `.obj` mesh files, `scipy` is required.

PETSIRD is on PyPI and conda-forge, so you can for instance do

```sh
python -m venv ~/petsirdvenv
. ~/petsirdvenv/bin/activate
git clone https://github.com/ETSIhackers/geom_visualisation.git
cd geom_visualisation
pip install -r requirements.txt
```

or

```sh
conda env create -f environment.yml
conda activate petsird-visualisation
```

## How to use

The script is located in the `python` folder. Try for instance

```sh
python -m petsird.helpers.generator | python python/main.py -o test.ply
```

The format of the output 3D mesh file is auto-detected by the extension. The supported formats are those supported by the `trimesh` python package.

# TODOs in the contribution

Enhance the viewer
  - Color hierachy module/detector
  - Checking if the geometry is okay
  - ~ Extract lut and save it
  - Use color choice
