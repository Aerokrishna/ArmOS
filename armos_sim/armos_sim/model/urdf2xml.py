#!/usr/bin/env python3

"""
Generic URDF -> MuJoCo MJCF converter.

Usage:
    python urdf_to_mjcf.py /path/to/robot.urdf

    python urdf_to_mjcf.py /path/to/robot.urdf -o /path/to/output

Requirements:
    pip install mujoco trimesh
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


SUPPORTED_MESH_FORMATS = {".stl", ".obj", ".dae", ".ply"}


def indent_xml(element: ET.Element, level: int = 0) -> None:
    """Pretty-print an ElementTree XML tree."""

    indentation = "\n" + level * "    "

    if len(element):
        if not element.text or not element.text.strip():
            element.text = indentation + "    "

        for child in element:
            indent_xml(child, level + 1)

        if not element.tail or not element.tail.strip():
            element.tail = indentation

    elif level and (not element.tail or not element.tail.strip()):
        element.tail = indentation

def resolve_mesh_path(mesh_filename: str, urdf_dir: Path) -> Path:
    """
    Resolve a mesh filename from a URDF.

    Searches for meshes starting one directory above the URDF directory.

    Handles:
        mesh.stl
        meshes/mesh.stl
        package://robot_description/meshes/mesh.stl
        file:///absolute/path/mesh.stl
    """

    mesh_filename = mesh_filename.strip()

    # Remove URI prefixes
    if mesh_filename.startswith("file://"):
        mesh_filename = mesh_filename[7:]

    if mesh_filename.startswith("package://"):
        # Remove package://package_name/
        remainder = mesh_filename[len("package://"):]
        parts = remainder.split("/", 1)

        if len(parts) == 2:
            mesh_filename = parts[1]
        else:
            mesh_filename = parts[0]

    mesh_path = Path(mesh_filename)

    # Absolute path
    if mesh_path.is_absolute() and mesh_path.exists():
        return mesh_path.resolve()

    # Search from one directory above the URDF directory
    search_root = urdf_dir.parent

    # Try the path relative to the parent directory
    candidate = (search_root / mesh_path).resolve()

    if candidate.exists():
        return candidate

    # Search recursively from the parent directory
    filename = mesh_path.name

    matches = list(search_root.rglob(filename))

    if matches:
        return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not find mesh referenced by URDF:\n"
        f"    {mesh_filename}\n"
        f"Searched recursively from:\n"
        f"    {search_root}"
    )

def convert_mesh_to_stl(source: Path, destination: Path) -> None:
    """Convert OBJ/DAE/PLY mesh to STL using trimesh."""

    import trimesh

    destination.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(str(source), force="mesh")

    if isinstance(mesh, trimesh.Scene):
        meshes = []

        for geometry in mesh.geometry.values():
            meshes.append(geometry)

        if not meshes:
            raise RuntimeError(f"No geometry found in mesh: {source}")

        mesh = trimesh.util.concatenate(meshes)

    mesh.export(str(destination))


def prepare_meshes(
    urdf_root: ET.Element,
    urdf_path: Path,
    output_dir: Path,
) -> None:
    """
    Copy/convert all meshes referenced by the URDF into:

        output_dir/meshes/

    and rewrite their URDF paths to be relative.
    """

    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    processed = {}

    for mesh in urdf_root.iter("mesh"):

        filename = mesh.get("filename")

        if not filename:
            continue

        source = resolve_mesh_path(filename, urdf_path.parent)

        # Preserve the original filename where possible.
        original_name = source.stem

        extension = source.suffix.lower()

        if extension not in SUPPORTED_MESH_FORMATS:
            raise RuntimeError(
                f"Unsupported mesh format '{extension}' for:\n"
                f"    {source}"
            )

        # MuJoCo handles STL very reliably.
        #
        # Convert OBJ/DAE/PLY to STL.
        if extension != ".stl":
            output_name = original_name + ".stl"
        else:
            output_name = source.name

        destination = meshes_dir / output_name

        # Deal with multiple meshes having the same filename.
        if output_name in processed:
            previous_source = processed[output_name]

            if previous_source != source:
                stem = source.stem
                output_name = f"{stem}_{abs(hash(str(source))) & 0xFFFF:04x}.stl"
                destination = meshes_dir / output_name

        if not destination.exists():

            if extension == ".stl":
                print(f"Copying mesh:     {source}")
                shutil.copy2(source, destination)

            else:
                print(f"Converting mesh:  {source}")
                print(f"               -> {destination}")

                convert_mesh_to_stl(source, destination)

        processed[output_name] = source

        # Make path relative to the generated MJCF.
        relative_mesh_path = Path("meshes") / output_name

        mesh.set(
            "filename",
            relative_mesh_path.as_posix(),
        )

    print(f"Prepared {len(processed)} mesh files.")


def fix_zero_inertias(root: ET.Element) -> None:
    """
    Replace zero diagonal inertia values.

    MuJoCo requires physically valid inertial parameters.
    This is disabled by default and can be enabled with --fix-inertia.
    """

    changed = 0

    for inertia in root.findall(".//inertia"):

        for attribute in ("ixx", "iyy", "izz"):

            value = float(inertia.get(attribute, "0"))

            if value <= 0.0:
                inertia.set(attribute, "1e-6")
                changed += 1

        for attribute in ("ixy", "ixz", "iyz"):

            if inertia.get(attribute) is None:
                inertia.set(attribute, "0.0")

    if changed:
        print(
            f"Fixed {changed} zero/invalid diagonal inertia values."
        )


def convert_urdf_to_mjcf(
    urdf_path: str | Path,
    output_dir: str | Path | None = None,
    fix_inertia: bool = False,
) -> Path:
    """
    Convert a URDF robot description into a MuJoCo MJCF XML.

    Parameters
    ----------
    urdf_path:
        Path to the input URDF.

    output_dir:
        Directory where the generated XML and meshes will be stored.
        Defaults to a directory next to the URDF.

    fix_inertia:
        Replace zero diagonal inertia values with a small positive value.

    Returns
    -------
    Path
        Path to the generated MJCF XML.
    """

    try:
        import mujoco
    except ImportError as exc:
        raise ImportError(
            "MuJoCo is required.\n"
            "Install with:\n"
            "    pip install mujoco"
        ) from exc

    urdf_path = Path(urdf_path).resolve()

    if not urdf_path.exists():
        raise FileNotFoundError(
            f"URDF file does not exist:\n    {urdf_path}"
        )

    if urdf_path.suffix.lower() != ".urdf":
        raise ValueError(
            f"Expected a .urdf file, got:\n    {urdf_path}"
        )

    if output_dir is None:
        output_dir = urdf_path.parent / f"{urdf_path.stem}_mjcf"

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("URDF -> MJCF")
    print("=" * 60)
    print(f"URDF:    {urdf_path}")
    print(f"Output:  {output_dir}")
    print()

    # ---------------------------------------------------------
    # 1. Read URDF
    # ---------------------------------------------------------

    tree = ET.parse(urdf_path)
    urdf_root = tree.getroot()

    if urdf_root.tag != "robot":
        raise ValueError(
            f"Expected URDF root element <robot>, "
            f"got <{urdf_root.tag}>"
        )

    robot_name = urdf_root.get("name", urdf_path.stem)

    print(f"Robot:   {robot_name}")

    # ---------------------------------------------------------
    # 2. Prepare meshes
    # ---------------------------------------------------------

    prepare_meshes(
        urdf_root,
        urdf_path,
        output_dir,
    )

    # ---------------------------------------------------------
    # 3. Optional inertia correction
    # ---------------------------------------------------------

    if fix_inertia:
        fix_zero_inertias(urdf_root)

    # ---------------------------------------------------------
    # 4. Add MuJoCo compiler settings
    # ---------------------------------------------------------

    # MuJoCo accepts <mujoco> inside a URDF and uses it
    # to configure the URDF -> MJCF conversion.

    mujoco_element = urdf_root.find("mujoco")

    if mujoco_element is None:
        mujoco_element = ET.Element("mujoco")
        urdf_root.insert(0, mujoco_element)

    compiler = mujoco_element.find("compiler")

    if compiler is None:
        compiler = ET.SubElement(
            mujoco_element,
            "compiler",
        )

    compiler.set("discardvisual", "false")
    compiler.set("fusestatic", "false")
    compiler.set("strippath", "false")

    # ---------------------------------------------------------
    # 5. Write temporary URDF
    # ---------------------------------------------------------

    temporary_urdf = output_dir / f"{urdf_path.stem}_prepared.urdf"

    tree.write(
        temporary_urdf,
        encoding="utf-8",
        xml_declaration=False,
    )

    print()
    print("Prepared URDF:")
    print(f"    {temporary_urdf}")

    # ---------------------------------------------------------
    # 6. Let MuJoCo compile URDF -> MJCF
    # ---------------------------------------------------------

    output_xml = output_dir / f"{urdf_path.stem}.xml"

    temporary_xml = None

    try:

        with tempfile.NamedTemporaryFile(
            suffix=".xml",
            delete=False,
        ) as tmp:
            temporary_xml = Path(tmp.name)

        print()
        print("Compiling URDF with MuJoCo...")

        model = mujoco.MjModel.from_xml_path(
            str(temporary_urdf)
        )

        # MuJoCo writes the fully compiled model as MJCF.
        mujoco.mj_saveLastXML(
            str(temporary_xml),
            model,
        )

        # -----------------------------------------------------
        # 7. Load generated XML and make it readable
        # -----------------------------------------------------

        mjcf_tree = ET.parse(temporary_xml)
        mjcf_root = mjcf_tree.getroot()

        indent_xml(mjcf_root)

        mjcf_tree.write(
            output_xml,
            encoding="utf-8",
            xml_declaration=False,
        )

    except Exception as exc:

        raise RuntimeError(
            "MuJoCo failed to convert the URDF.\n\n"
            f"URDF:\n    {urdf_path}\n\n"
            f"Prepared URDF:\n    {temporary_urdf}\n\n"
            f"Original error:\n    {exc}"
        ) from exc

    finally:

        if temporary_xml is not None:
            temporary_xml.unlink(missing_ok=True)

        temporary_urdf.unlink(missing_ok=True)

    print()
    print("=" * 60)
    print("Conversion successful!")
    print("=" * 60)
    print(f"MJCF:    {output_xml}")
    print(f"Meshes:  {output_dir / 'meshes'}")
    print()

    return output_xml


def main() -> None:

    parser = argparse.ArgumentParser(
        description="Convert a URDF robot description to MuJoCo MJCF."
    )

    parser.add_argument(
        "urdf",
        type=Path,
        help="Path to the input URDF file.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output directory. "
            "Defaults to <urdf_name>_mjcf next to the URDF."
        ),
    )

    parser.add_argument(
        "--fix-inertia",
        action="store_true",
        help="Replace zero diagonal inertia values with a small positive value.",
    )

    args = parser.parse_args()

    convert_urdf_to_mjcf(
        urdf_path=args.urdf,
        output_dir=args.output,
        fix_inertia=args.fix_inertia,
    )


if __name__ == "__main__":
    main()