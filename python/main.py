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
import trimesh
import argparse


#########################################################################################
# Constants
#########################################################################################
crystal_color = np.array([255, 40, 40], dtype=np.uint8)


#########################################################################################
# Methods
#########################################################################################
def transform_to_mat44(
    transform: petsird.RigidTransformation,
) -> npt.NDArray[np.float32]:
    return np.vstack([transform.matrix, [0, 0, 0, 1]])


def mat44_to_transform(mat: npt.NDArray[np.float32]) -> petsird.RigidTransformation:
    return petsird.RigidTransformation(matrix=mat[0:3, :])


def coordinate_to_homogeneous(coord: petsird.Coordinate) -> npt.NDArray[np.float32]:
    return np.hstack([coord.c, 1])


def homogeneous_to_coordinate(
    hom_coord: npt.NDArray[np.float32],
) -> petsird.Coordinate:
    return petsird.Coordinate(c=hom_coord[0:3])


def mult_transforms(
    transforms: list[petsird.RigidTransformation],
) -> petsird.RigidTransformation:
    """multiply rigid transformations"""
    mat = np.array(
        ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)),
        dtype="float32",
    )

    for t in reversed(transforms):
        mat = np.matmul(transform_to_mat44(t), mat)
    return mat44_to_transform(mat)


def mult_transforms_coord(
    transforms: list[petsird.RigidTransformation], coord: petsird.Coordinate
) -> petsird.Coordinate:
    """apply list of transformations to coordinate"""
    # TODO better to multiply with coordinates in sequence, as first multiplying the matrices
    hom = np.matmul(
        transform_to_mat44(mult_transforms(transforms)),
        coordinate_to_homogeneous(coord),
    )
    return homogeneous_to_coordinate(hom)


def transform_BoxShape(
    transform: petsird.RigidTransformation, box_shape: petsird.BoxShape
) -> petsird.BoxShape:
    return petsird.BoxShape(
        corners=[mult_transforms_coord([transform], c) for c in box_shape.corners]
    )


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

    if header.scanner.detection_efficiencies.det_el_efficiencies is None:
        raise ValueError(
            "The scanner detection efficiencies are not defined. Correct this or remove the show_det_eff flag."
        )
    return header.scanner.detection_efficiencies.det_el_efficiencies

    
def set_detector_color(det_mesh, detector_efficiencies, mod_i, num_det_in_module, det_i, random_color):
    if random_color == True:
        color = np.random.randint(0, 255, size=3)
    elif detector_efficiencies is not None:
        color = (crystal_color * detector_efficiencies[mod_i * num_det_in_module + det_i])
    else:
        color = crystal_color
    
    f_color = np.array([color[0], color[1], color[2], 50]).astype(
        np.uint8
    )

    det_mesh.visual.face_colors = f_color

    return det_mesh



def set_module_color(
    module_mesh, detector_efficiencies, mod_i, num_det_in_module, det_el, random_color
):
    if random_color == True:
        color = np.random.randint(0, 255, size=3)
    elif detector_efficiencies is not None:
        # Mean of the detector efficiency in the current module
        color = crystal_color * np.mean(
            detector_efficiencies.reshape((-1, len(det_el) * num_det_in_module))[
                mod_i, :
            ]
        )
    else:
        color = crystal_color

    f_color = np.array([color[0], color[1], color[2], 50]).astype(np.uint8)

    module_mesh.visual.face_colors = f_color

    return module_mesh


def create_mesh(header, modules_only=False, show_det_eff=False, random_color=False, fov=None,
                start_modules=0, end_modules=-1, step_modules=1):
    """
    Create 3D model of PET scanner defined in PETSIRD list mode file.
    """
    detector_efficiencies = extract_detector_eff(show_det_eff, header)

    shapes = []
    # draw all crystals
    for rep_module in header.scanner.scanner_geometry.replicated_modules:
        det_el = (
            rep_module.object.detecting_elements
        )  # Get all the detecting elements modules
        max_num_modules = len(rep_module.transforms)
        end_range = max_num_modules if (end_modules < 0 or end_modules > max_num_modules) else end_modules
        for mod_i in range(start_modules, end_range, step_modules):

            vertices = []  # If showing modules only
            mod_transform = rep_module.transforms[mod_i]
            for rep_volume in det_el:
                num_det_in_module = len(rep_volume.transforms)
                for det_i in range(num_det_in_module):
                    transform = rep_volume.transforms[det_i]
                    box: petsird.BoxShape = transform_BoxShape(
                        mult_transforms([mod_transform, transform]),
                        rep_volume.object.shape,
                    )
                    corners = []
                    for boxcorner in box.corners:
                        corners.append(boxcorner.c)
                        
                    if not modules_only:
                        det_mesh = create_box_from_vertices(corners)
                            
                        det_mesh = set_detector_color(det_mesh, detector_efficiencies, mod_i, num_det_in_module, det_i, random_color)
                            
                        shapes.append(det_mesh)
                    else:
                        vertices.append(corners)
            if modules_only:
                vertices_reshaped = np.array(vertices).reshape((-1, 3))
                module_mesh = trimesh.convex.convex_hull(vertices_reshaped)

                module_mesh = set_module_color(
                    module_mesh,
                    detector_efficiencies,
                    mod_i,
                    num_det_in_module,
                    det_el, random_color
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
        "--start-modules",
        type=int,
        dest="start_modules",
        default=0,
        required=False,
        help="Start index of the modules to show. Example: start-modules=6 will display modules 6,7,8,...",
    )
    parser.add_argument(
        "--end-modules",
        type=int,
        dest="end_modules",
        default=-1,
        required=False,
        help="End index of the modules to show. Example: end-modules=16 will display modules ...,14,15,16",
    )
    parser.add_argument(
        "--step-modules",
        type=int,
        dest="step_modules",
        default=1,
        required=False,
        help="Incrementation of the module indices to show. Example: step-modules=4 will display modules 0,4,8,12,...",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parserCreator()

    file = None
    if args.input is None:
        file = sys.stdin.buffer
    else:
        file = open(args.input, "rb")
    output_fname = args.output

    with petsird.BinaryPETSIRDReader(file) as reader:
        header = reader.read_header()

        mesh = create_mesh(header, args.modules_only, args.show_det_eff, args.random_color, args.fov,
                           args.start_modules, args.end_modules, args.step_modules)
        mesh.export(output_fname)

        # Forced to do this
        for time_block in reader.read_time_blocks():
            pass