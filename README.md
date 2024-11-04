# Description
This module reads a PETSIRD data file and generates a 3D mesh file.

## How to use

The script is located in the `python` folder.

Additional dependency: `trimesh`. If generating `.obj` mesh files, `scipy` is required.

The format of the output 3D mesh file is auto-detected by the extension. The supported formats are those supported by the `trimesh` python package.

# TODOs in the contribution
Enhance the viewer
  - Color hierachy module/detector
  - Checking if the geometry is okay
  - ~ Extract lut and save it
