#!/bin/env python
import sys
import os
import numpy as np
import numpy.typing as npt
import petsird
import trimesh
import argparse


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


def create_box_from_vertices(vertices, color=None):
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
    if color is not None:
        # v_color = np.array([color[0],color[1],color[2], 50] * len(vertices)).astype(np.uint8)
        # box.visual.vertex_colors = v_color
        f_color = np.array([color[0], color[1], color[2], 50]).astype(np.uint8)
        box.visual.face_colors = f_color
    return box


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read from a file or stdin.")
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
        help="File to write",
    )
    parser.add_argument(
        "--fov",
        type=float,
        nargs=2,
        default=None,
        required=False,
        help="Add a cylindrical FOV, as radius and height",
    )
    args = parser.parse_args()

    file = None
    if args.input is None:
        file = sys.stdin.buffer
    else:
        file = open(args.input, "rb")
    output_fname = args.output

    with petsird.BinaryPETSIRDReader(file) as reader:
        header = reader.read_header()

        # Forced to do this
        for time_block in reader.read_time_blocks():
            pass

        crystal_color = np.array([255, 40, 40], dtype=np.uint8)

        detector_efficiencies = (
            header.scanner.detection_efficiencies.det_el_efficiencies
        )
        detector_efficiencies = np.mean(detector_efficiencies, axis=1)

        shapes = []
        # draw all crystals
        for rep_module in header.scanner.scanner_geometry.replicated_modules:
            det_el = (
                rep_module.object.detecting_elements
            )  # Get all the detecting elements
            for mod_i in range(
                len(rep_module.transforms)
            ):  # For each transformation of the module
                mod_transform = rep_module.transforms[mod_i]
                for rep_volume in det_el:  # For each detector in the module
                    num_det_in_module = len(rep_volume.transforms)
                    for det_i in range(
                        num_det_in_module
                    ):  # For each transformation in the detector
                        transform = rep_volume.transforms[det_i]
                        box: petsird.BoxShape = transform_BoxShape(
                            mult_transforms([mod_transform, transform]),
                            rep_volume.object.shape,
                        )
                        corners = []
                        for boxcorner in box.corners:
                            corners.append(boxcorner.c)
                        color = (
                            crystal_color
                            * detector_efficiencies[mod_i * num_det_in_module + det_i]
                        )
                        shapes.append(create_box_from_vertices(corners, color))
        if args.fov is not None:
            shapes.append(
                trimesh.creation.cylinder(radius=args.fov[0], height=args.fov[1])
            )
        combined = trimesh.util.concatenate(shapes)
        combined.export(output_fname)
