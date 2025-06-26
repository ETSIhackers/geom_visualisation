#!/usr/bin/env python

"""
Create 3D model file of PET scanner defined in PETSIRD list mode file.
Known format that work:
    - ply, stl, obj
With color? 
    - ply

3D viewer tried:
    meshlab: Work with color 
    blender: Voodoo magic for color?
    paraview: Work with color sometime?

Potential issues:
    - If the number of detectors is not the same for each type of detectors, color stuff will break.

"""


#########################################################################################
# Import
#########################################################################################
import sys
import os
import numpy as np
import numpy.typing as npt
import petsird
import petsird.helpers.geometry
import trimesh
import argparse


#########################################################################################
# Constants
#########################################################################################
crystal_color = np.array([255, 40, 40], dtype=np.uint8)


#########################################################################################
# Methods
#########################################################################################

def create_box_from_vertices(vertices):
    # Define faces using the indices of vertices that make up each face
    faces = [
        [1, 0, 2],
        [0, 2, 3],  # Bottom face
        [4, 5, 6],
        [4, 6, 7],  # Top face
        [0, 3, 7],
        [0, 7, 4],  # Left face
        [1, 2, 6],
        [1, 6, 5],  # Right face
        [0, 1, 5],
        [0, 5, 4],  # Front face
        [3, 2, 6],
        [3, 6, 7],  # Back face
    ]

    # Create and return a Trimesh object
    box = trimesh.Trimesh(vertices=vertices, faces=faces)

    return box


def extract_detector_eff(show_det_eff, header):
    if not show_det_eff:
        return None

    eff = header.scanner.detection_efficiencies.detection_bin_efficiencies
    if eff is None:
        raise ValueError(
            "The scanner detection efficiencies are not defined. Correct this or remove the show_det_eff flag."
        )
    return eff

    
def set_detector_color(det_mesh, detector_efficiencies, type_of_module, num_modules, mod_i, num_det_in_module, det_i, random_color):
    if random_color == True:
        color = np.random.randint(0, 255, size=3)
    elif detector_efficiencies is not None:
        eff = np.reshape(detector_efficiencies[type_of_module], shape=(num_modules, num_det_in_module, -1))
        color = (crystal_color * np.mean(eff[mod_i, det_i, :]))
    else:
        color = crystal_color
    
    f_color = np.array([color[0], color[1], color[2], 50]).astype(
        np.uint8
    )
    print(f_color)
    det_mesh.visual.face_colors = f_color

    return det_mesh



def set_module_color(
        module_mesh, detector_efficiencies, type_of_module, num_modules, mod_i, num_det_in_module, random_color
):
    if random_color == True:
        color = np.random.randint(0, 255, size=3)
    elif detector_efficiencies is not None:
        # Mean of the detector efficiencies in the current module
        eff = np.reshape(detector_efficiencies[type_of_module], shape=(num_modules, num_det_in_module, -1))
        color = crystal_color * np.mean(
            eff[mod_i, :, :])
    else:
        color = crystal_color

    f_color = np.array([color[0], color[1], color[2], 50]).astype(np.uint8)

    module_mesh.visual.face_colors = f_color

    return module_mesh


def create_mesh(header, modules_only=False, show_det_eff=False, random_color=False, fov=None,
                modules_indices=None):
    """
    Create 3D model of PET scanner defined in PETSIRD list mode file.
    """
    detector_efficiencies = extract_detector_eff(show_det_eff, header)

    shapes = []
    # draw all crystals
    module_count = 0
    for type_of_module in range(len(header.scanner.scanner_geometry.replicated_modules)):
        rep_module = header.scanner.scanner_geometry.replicated_modules[type_of_module]
        num_modules = len(rep_module.transforms)
        det_els = (
            rep_module.object.detecting_elements
        )  # Get all the detecting elements modules
        for mod_i in range(len(rep_module.transforms)):
            module_count += 1
            if modules_indices is not None and (module_count - 1) not in modules_indices:
                continue

            vertices = []  # If showing modules only
            mod_transform = rep_module.transforms[mod_i]
            num_det_in_module = len(det_els.transforms)
            for det_i in range(num_det_in_module):
                transform = det_els.transforms[det_i]
                box: petsird.BoxShape = petsird.helpers.geometry.transform_BoxShape(
                    petsird.helpers.geometry.mult_transforms(
                        [mod_transform, transform]),
                    det_els.object.shape,
                )
                corners = []
                for boxcorner in box.corners:
                    corners.append(boxcorner.c)

                if not modules_only:
                    det_mesh = create_box_from_vertices(corners)
                            
                    det_mesh = set_detector_color(det_mesh, detector_efficiencies, type_of_module, num_modules, mod_i, num_det_in_module, det_i, random_color)
                            
                    shapes.append(det_mesh)
                else:
                    vertices.append(corners)
            if modules_only:
                vertices_reshaped = np.array(vertices).reshape((-1, 3))
                module_mesh = trimesh.convex.convex_hull(vertices_reshaped)

                module_mesh = set_module_color(
                    module_mesh,
                    detector_efficiencies,
                    type_of_module,
                    num_modules,
                    mod_i,
                    num_det_in_module,
                    random_color
                )

                shapes.append(module_mesh)

    if fov is not None:
        shapes.append(
            trimesh.creation.cylinder(radius=fov[0], height=fov[1])
        )
    return trimesh.util.concatenate(shapes)


#########################################################################################
# Main
#########################################################################################
def parserCreator():
    parser = argparse.ArgumentParser(description="Create 3D model of PET scanner defined in PETSIRD list mode file.")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="File to read from, or stdin if omitted",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=True,
        help="File to write (with the appropriate file format extension)",
    )
    parser.add_argument(
        "--fov",
        type=float,
        nargs=2,
        default=None,
        required=False,
        help="Add a cylindrical FOV, as radius and height",
    )
    parser.add_argument(
        "--modules-only",
        action="store_true",
        dest="modules_only",
        default=False,
        required=False,
        help="Generate shapes for modules only",
    )
    parser.add_argument(
        "--show-det-eff",
        action="store_true",
        dest="show_det_eff",
        default=False,
        required=False,
        help="Change color following detector effeciency",
    )
    parser.add_argument(
        "--random-color",
        action="store_true",
        dest="random_color",
        default=False,
        required=False,
        help="Force random color for easier debug position",
    )
    parser.add_argument(
        "--num-modules",
        type=int,
        dest="num_modules",
        default=0,
        required=False,
        help="Max number of modules to convert",
    )
    parser.add_argument(
        "--skip-modules",
        type=int,
        dest="skip_modules",
        default=0,
        required=False,
        help="Skip every xx modules",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parserCreator()
    if args.skip_modules < 0:
        sys.exit('skip_modules should be >= 0')

    file = None
    if args.input is None:
        file = sys.stdin.buffer
    else:
        file = open(args.input, "rb")
        output_fname = args.output

    with petsird.BinaryPETSIRDReader(file) as reader:
        header = reader.read_header()
        module_indices = range(0, args.num_modules * (args.skip_modules + 1), args.skip_modules + 1) if args.num_modules > 0 else None
        mesh = create_mesh(header, args.modules_only, args.show_det_eff, args.random_color, args.fov,
                          module_indices
        )
        mesh.export(output_fname)

        # Forced to do this
        for time_block in reader.read_time_blocks():
            pass
