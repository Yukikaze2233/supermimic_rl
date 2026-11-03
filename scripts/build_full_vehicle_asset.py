#!/usr/bin/env python3
"""Prepare an audited full-vehicle candidate around a historical wheel-leg core.

This is an offline CPU asset conversion, not a physics qualification. Historical
core joint coordinates, six loop anchors and gas-spring data are preserved until
the new CAD travel review supplies their replacement. CAD inertias are transported
into those coordinates. Auxiliary gimbal joints remain outside the q18 reset model.
The pitch range is user-specified native travel, not a measured physical stop.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation
import trimesh

AUX_JOINTS = ["M_joint1", "M_joint2"]
AUX_BODIES = ["M_link1", "M_link2"]
REVISION = "full_vehicle_v2p0p1_source_audited_v2"
URDF_FORMAT_VERSION = "1.2"
PITCH_LIMITS_DEG = (-30.0, 35.0)
ROOT_ROTATION = np.eye(4)
ROOT_ROTATION[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
# Source-CAD empty spaces preserved explicitly after bounded convex decomposition.
CLEARANCE_VOIDS = {
    "base_link": [
        ([-.06, -.30, -.20], [.06, .30, -.15]),
        ([-.06, .15, -.20], [.06, .30, -.125]),
        ([-.06, -.30, -.20], [.06, -.15, -.125]),
    ],
    "M_link1": [([.07, -.035, .025], [.15, .015, .05])],
    "M_link2": [],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def origin(node: ET.Element | None) -> np.ndarray:
    matrix = np.eye(4)
    if node is not None:
        matrix[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(node.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        matrix[:3, 3] = np.fromstring(node.get("xyz", "0 0 0"), sep=" ")
    return matrix


def xml_joints(root: ET.Element) -> list[dict]:
    return [dict(name=j.get("name"), parent=j.find("parent").get("link"),
                 child=j.find("child").get("link"), type=j.get("type"),
                 origin=origin(j.find("origin")).tolist(),
                 axis=np.fromstring(j.find("axis").get("xyz"), sep=" ").tolist())
            for j in root.findall("joint")]


def fk(joints: list[dict], q: dict | None = None) -> dict[str, np.ndarray]:
    q = q or {}
    poses = {"base_link": np.eye(4)}
    remaining = list(joints)
    while remaining:
        resolved = [j for j in remaining if j["parent"] in poses]
        if not resolved:
            raise ValueError("Disconnected or cyclic joint tree")
        for joint in resolved:
            motion = np.eye(4)
            axis = np.asarray(joint["axis"], dtype=float)
            angle = q.get(joint["name"], 0.0)
            if joint["type"] == "prismatic":
                motion[:3, 3] = axis * angle
            elif joint["type"] != "fixed":
                motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            poses[joint["child"]] = poses[joint["parent"]] @ joint["origin"] @ motion
            remaining.remove(joint)
    return poses


def inertia(link: ET.Element, mapping: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    data = link.find("inertial")
    frame = mapping @ origin(data.find("origin"))
    values = {k: float(v) for k, v in data.find("inertia").attrib.items()}
    tensor = np.array([[values["ixx"], values["ixy"], values["ixz"]],
                       [values["ixy"], values["iyy"], values["iyz"]],
                       [values["ixz"], values["iyz"], values["izz"]]])
    # URDF inertia is about its COM, so translation does not add a parallel-axis term.
    tensor = frame[:3, :3] @ tensor @ frame[:3, :3].T
    eigenvalues = np.linalg.eigvalsh(tensor)
    if eigenvalues[0] <= 0 or eigenvalues[0] + eigenvalues[1] < eigenvalues[2] - 1e-12:
        raise ValueError(f"Invalid source inertia: {link.get('name')}")
    return float(data.find("mass").get("value")), frame[:3, 3], tensor


def xml_origin(parent: ET.Element, matrix: np.ndarray) -> None:
    ET.SubElement(parent, "origin", xyz=numbers(matrix[:3, 3]),
                  rpy=numbers(Rotation.from_matrix(matrix[:3, :3]).as_euler("xyz")))


def numbers(values) -> str:
    return " ".join(format(float(x), ".17g") for x in values)


def configure_gimbal_pitch(gimbal: dict) -> dict:
    """Bind user travel in native coordinates after checking the optical FK sign."""
    result = copy.deepcopy(gimbal)
    pitch = next(joint for joint in result["joints"] if joint["name"] == "M_joint2")
    bounds = np.deg2rad(PITCH_LIMITS_DEG).tolist()
    pitch["type"] = "revolute"
    pitch["mechanical_position_limits_rad"] = bounds
    pitch["position_limit_source"] = "user_specified_20261101_down30_up35_not_measured"
    pitch["limit"] = {"lower": bounds[0], "upper": bounds[1],
                      "effort": pitch["effort_limit_nm"]}
    # A missing measured speed remains missing; do not invent one to satisfy a
    # generic URDF importer. This bundle's training path reads the authored USD.
    result["mechanical_position_limits_status"] = "pitch_user_specified_not_measured; yaw_unknown"
    result["pitch_position_contract"] = {
        "joint_name": "M_joint2", "native_limits_deg": list(PITCH_LIMITS_DEG),
        "reference": "source zero; positive native pitch raises the view at zero yaw",
        "source": "user_specified_20261101", "measured_stop": False,
        "control_signs_are_separate": True,
    }
    result["pitch_position_contract"]["fk_elevation_check"] = pitch_elevation_check(result)
    return result


def pitch_elevation_check(gimbal: dict) -> dict:
    """Verify native pitch direction using the actual view-body forward vector."""
    samples = {}
    for degrees in (PITCH_LIMITS_DEG[0], -1.0, 0.0, 1.0, PITCH_LIMITS_DEG[1]):
        pose = fk(gimbal["joints"], {"M_joint1": 0., "M_joint2": np.deg2rad(degrees)})
        direction = pose[gimbal["view_body"]][:3, :3] @ gimbal["view_forward_local"]
        elevation = float(np.rad2deg(np.arctan2(direction[2], np.linalg.norm(direction[:2]))))
        samples[str(degrees)] = {"view_forward_control": direction.tolist(), "elevation_deg": elevation}
    baseline = samples["0.0"]["elevation_deg"]
    lower, upper = (samples[str(bound)]["elevation_deg"] for bound in PITCH_LIMITS_DEG)
    if not (lower < samples["-1.0"]["elevation_deg"] < baseline
            < samples["1.0"]["elevation_deg"] < upper):
        raise ValueError("native M_joint2 sign no longer means down-negative/up-positive in optical FK")
    if max(abs((lower - baseline) - PITCH_LIMITS_DEG[0]),
           abs((upper - baseline) - PITCH_LIMITS_DEG[1])) > .01:
        raise ValueError("native pitch travel differs from source-zero view elevation by more than .01 degree")
    return {"yaw_native_rad": 0., "samples": samples,
            "baseline_elevation_deg": baseline,
            "maximum_endpoint_delta_from_native_deg": max(
                abs((lower - baseline) - PITCH_LIMITS_DEG[0]),
                abs((upper - baseline) - PITCH_LIMITS_DEG[1]))}


def validate_renderable_visuals(stage, body_names: list[str]) -> dict:
    """Reject references whose stronger Xform type masks a source Mesh."""
    from pxr import Usd, UsdGeom

    result = {}
    for name in body_names:
        visual = stage.GetPrimAtPath("/Robot/" + name + "/Visual")
        meshes = [prim for prim in Usd.PrimRange(visual) if prim.IsA(UsdGeom.Mesh)] if visual else []
        if len(meshes) != 1:
            raise ValueError(f"Expected one renderable visual Mesh for {name}; found {len(meshes)}")
        mesh = UsdGeom.Mesh(meshes[0])
        points = mesh.GetPointsAttr().Get()
        counts = mesh.GetFaceVertexCountsAttr().Get()
        indices = mesh.GetFaceVertexIndicesAttr().Get()
        if not points or not counts or not indices or sum(counts) != len(indices):
            raise ValueError(f"Renderable visual has empty or inconsistent geometry: {name}")
        result[name] = {"path": str(meshes[0].GetPath()), "points": len(points), "faces": len(counts)}
    return result


def clip_mesh(mesh: trimesh.Trimesh, normal: np.ndarray, offset: float) -> trimesh.Trimesh:
    vertices, faces, _ = trimesh.intersections.slice_faces_plane(
        mesh.vertices, mesh.faces, normal, normal * offset)
    return trimesh.Trimesh(vertices, faces, process=False)


def preserve_clearance(mesh: trimesh.Trimesh, hulls: list[trimesh.Trimesh],
                       name: str) -> tuple[list[trimesh.Trimesh], list[dict]]:
    """Subtract only CAD-empty boxes; keep six disjoint convex outside pieces."""
    evidence = []
    for lower, upper in CLEARANCE_VOIDS[name]:
        source_piece = mesh
        for axis in range(3):
            normal = np.eye(3)[axis]
            source_piece = clip_mesh(source_piece, normal, lower[axis])
            source_piece = clip_mesh(source_piece, -normal, -upper[axis])
        if len(source_piece.faces):
            raise ValueError(f"Clearance region intersects source CAD: {name}")
        result = []
        for hull in hulls:
            if np.any(hull.bounds[1] <= lower) or np.any(hull.bounds[0] >= upper):
                result.append(hull)
                continue
            remaining = hull
            for axis in range(3):
                normal = np.eye(3)[axis]
                for outside_normal, outside_offset, inside_normal, inside_offset in (
                    (-normal, -lower[axis], normal, lower[axis]),
                    (normal, upper[axis], -normal, -upper[axis]),
                ):
                    if not len(remaining.faces):
                        break
                    outside = clip_mesh(remaining, outside_normal, outside_offset)
                    if len(outside.faces) and np.all(outside.extents > 1e-8):
                        result.append(outside.convex_hull)
                    remaining = clip_mesh(remaining, inside_normal, inside_offset)
                    if len(remaining.faces) and np.all(remaining.extents > 1e-8):
                        remaining = remaining.convex_hull
                    else:
                        remaining = trimesh.Trimesh()
        evidence.append(dict(lower_source_m=lower, upper_source_m=upper,
                             source_intersecting_triangles=0, hulls_before=len(hulls),
                             hulls_after=len(result)))
        hulls = result
    return hulls, evidence


def make_collisions(mesh: trimesh.Trimesh, name: str, mapping: np.ndarray,
                    out: Path, settings: dict) -> tuple[list[dict], dict]:
    import coacd

    coacd.set_log_level("warn")
    pieces = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **settings)
    hull_meshes = []
    discarded_degenerate_pieces = 0
    for vertices, faces in pieces:
        vertices = np.asarray(vertices)
        # A zero-volume fragment is not a physical contact shape. Avoid a
        # Qhull failure on a preprocessing sliver rather than inventing volume.
        if np.linalg.matrix_rank(vertices - vertices.mean(axis=0), tol=1e-10) < 3:
            discarded_degenerate_pieces += 1
            continue
        hull_meshes.append(trimesh.Trimesh(vertices, faces, process=True).convex_hull)
    hull_meshes, clearance_evidence = preserve_clearance(mesh, hull_meshes, name)
    collisions = []
    evidence = []
    probes = mesh.vertices[::max(1, len(mesh.vertices) // 4096)][:4096]
    outside_lower_bound = np.full(len(probes), np.inf)
    source_vertices = cKDTree(mesh.vertices)
    proxy_vertex_distances = []
    for index, hull in enumerate(hull_meshes):
        method = "coacd_convex_hull"
        # PhysX convex meshes have a bounded vertex budget.
        if len(hull.vertices) > 255:
            # An OBB after clearance carving would fill the same gap again.
            # Inscribed support samples preserve the empty halfspaces.
            directions = trimesh.creation.icosphere(subdivisions=2).vertices
            indices = np.unique(np.argmax(hull.vertices @ directions.T, axis=0))
            hull = trimesh.Trimesh(vertices=hull.vertices[indices], faces=[]).convex_hull
            method = "coacd_hull_inscribed_support_vertex_limit"
        relative = f"collisions/{name}_full_{index:03d}.obj"
        hull.export(out / relative)
        planes = ConvexHull(hull.vertices).equations
        violation = np.maximum(0., (probes @ planes[:, :3].T + planes[:, 3]).max(axis=1))
        outside_lower_bound = np.minimum(outside_lower_bound, violation)
        proxy_vertex_distances.extend(source_vertices.query(hull.vertices)[0].tolist())
        collisions.append(dict(type="mesh", file=relative, origin=mapping.tolist()))
        evidence.append(dict(file=relative, vertices=len(hull.vertices),
                             faces=len(hull.faces), volume_m3=float(abs(hull.volume)),
                             method=method))
    if not collisions:
        raise RuntimeError(f"No collision proxy produced for {name}")
    return collisions, dict(method="coacd_with_CAD_clearance_carving", settings=settings,
                            initial_hull_budget=settings["max_convex_hull"],
                            discarded_degenerate_pieces=discarded_degenerate_pieces,
                            protected_cad_empty_clearances=clearance_evidence,
                            source_faces=len(mesh.faces), source_vertices=len(mesh.vertices),
                            source_bounds_m=mesh.bounds.tolist(), source_watertight=mesh.is_watertight,
                            source_signed_volume_m3=float(mesh.volume), hulls=evidence,
                            summed_hull_volume_m3=sum(x["volume_m3"] for x in evidence),
                            source_probe_count=len(probes),
                            source_probe_convex_union_halfspace_violation_max_m=float(outside_lower_bound.max()),
                            proxy_vertex_to_source_vertex_max_m=float(max(proxy_vertex_distances)),
                            proxy_vertex_to_source_vertex_p95_m=float(np.quantile(proxy_vertex_distances, .95)),
                            distance_scope="Deterministic vertex/halfspace diagnostics, not a surface Hausdorff bound.",
                            note="Hull sums include overlaps; not a volume error or contact qualification.")


def write_urdf(old: Path, out: Path, bodies: list[dict], gimbal: dict) -> None:
    root = ET.parse(old / "robot.urdf").getroot()
    root.set("name", REVISION)
    # URDF 1.2 permits an unknown velocity bound to remain omitted. Older
    # formats require this field; omitting the version would select 1.0.
    root.set("version", URDF_FORMAT_VERSION)
    for body in bodies:
        name = body["name"]
        link = root.find(f"link[@name='{name}']")
        if link is None:
            link = ET.SubElement(root, "link", name=name)
        for item in list(link):
            if item.tag in ("inertial", "visual") or (item.tag == "collision" and name in ("base_link", *AUX_BODIES)):
                link.remove(item)
        inertial = ET.SubElement(link, "inertial")
        frame = np.eye(4)
        frame[:3, 3] = body["com"]
        xml_origin(inertial, frame)
        ET.SubElement(inertial, "mass", value=str(body["mass"]))
        tensor = np.asarray(body["inertia"])
        ET.SubElement(inertial, "inertia", **{key: format(tensor[i, j], ".17g")
                      for key, i, j in [("ixx", 0, 0), ("ixy", 0, 1), ("ixz", 0, 2),
                                        ("iyy", 1, 1), ("iyz", 1, 2), ("izz", 2, 2)]})
        visual = ET.SubElement(link, "visual")
        xml_origin(visual, np.asarray(body["mesh_origin"]))
        ET.SubElement(ET.SubElement(visual, "geometry"), "mesh", filename=body["mesh"])
        material = ET.SubElement(visual, "material", name=name)
        ET.SubElement(material, "color", rgba=numbers([*body["color"], 1]))
        if name in ("base_link", *AUX_BODIES):
            for index, collision in enumerate(body["collisions"]):
                node = ET.SubElement(link, "collision", name=f"{name}_full_{index:03d}")
                xml_origin(node, np.asarray(collision["origin"]))
                ET.SubElement(ET.SubElement(node, "geometry"), "mesh", filename=collision["file"])
    for joint in gimbal["joints"]:
        node = ET.SubElement(root, "joint", name=joint["name"], type=joint["type"])
        ET.SubElement(node, "parent", link=joint["parent"])
        ET.SubElement(node, "child", link=joint["child"])
        xml_origin(node, np.asarray(joint["origin"]))
        ET.SubElement(node, "axis", xyz=numbers(joint["axis"]))
        if joint["type"] == "revolute":
            bounds = joint["mechanical_position_limits_rad"]
            attributes = {"lower": numbers([bounds[0]]), "upper": numbers([bounds[1]]),
                          "effort": numbers([joint["effort_limit_nm"]])}
            if joint.get("velocity_limit_rad_s") is not None:
                attributes["velocity"] = numbers([joint["velocity_limit_rad_s"]])
            ET.SubElement(node, "limit", **attributes)
        # Unknown physical speed and yaw stops remain unspecified.
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(out / "robot.urdf", encoding="utf-8", xml_declaration=True)


def write_usd(old: Path, out: Path, bodies: list[dict], gimbal: dict) -> None:
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

    def quat(matrix, single=False):
        q = Rotation.from_matrix(matrix[:3, :3]).as_quat()
        return (Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])) if single
                else Gf.Quatd(float(q[3]), Gf.Vec3d(*q[:3])))

    def pose(prim, matrix):
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*matrix[:3, 3]))
        xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(quat(matrix))

    def mesh_prim(stage, path, mesh):
        node = UsdGeom.Mesh.Define(stage, path)
        node.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(mesh.vertices, dtype=np.float32)))
        node.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(mesh.faces), 3, np.int32)))
        node.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(mesh.faces, dtype=np.int32).ravel()))
        node.CreateSubdivisionSchemeAttr("none")
        return node

    # Preserve calibrated joint frame/limit authoring and all six external constraints.
    parent_stage = Usd.Stage.Open(str(old / "robot.usda"))
    parent_stage.GetRootLayer().Export(str(out / "robot.usda"))
    stage = Usd.Stage.Open(str(out / "robot.usda"))
    visual_layer = out / "meshes/full_vehicle_visuals.usdc"
    visual_layer.unlink(missing_ok=True)
    visuals = Usd.Stage.CreateNew(str(visual_layer))
    visuals.SetDefaultPrim(UsdGeom.Xform.Define(visuals, "/Visuals").GetPrim())
    auxiliary_poses = fk(gimbal["joints"])
    base_pose = np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath("/Robot/base_link")).GetLocalTransformation()).T
    for body in bodies:
        name = body["name"]
        path = "/Robot/" + name
        if name in AUX_BODIES:
            prim = UsdGeom.Xform.Define(stage, path).GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim).CreateRigidBodyEnabledAttr(True)
            pose(prim, base_pose @ auxiliary_poses[name])
        else:
            prim = stage.GetPrimAtPath(path)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(body["mass"])
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*body["com"]))
        eigenvalues, axes = np.linalg.eigh(body["inertia"])
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*eigenvalues))
        mass.CreatePrincipalAxesAttr(quat(axes, single=True))
        mesh = trimesh.load(out / body["mesh"], force="mesh", process=False)
        visual = mesh_prim(visuals, "/Visuals/" + name, mesh)
        visual.CreateDisplayColorAttr([Gf.Vec3f(*body["color"])])
        stage.RemovePrim(path + "/Visual")
        # A local Xform declaration overrides the referenced Mesh's type and
        # makes its points non-renderable. Mesh also supports the same pose ops.
        visible = UsdGeom.Mesh.Define(stage, path + "/Visual").GetPrim()
        visible.GetReferences().AddReference("meshes/full_vehicle_visuals.usdc", "/Visuals/" + name)
        pose(visible, np.asarray(body["mesh_origin"]))
        if name in ("base_link", *AUX_BODIES):
            for child in list(prim.GetChildren()):
                if child.HasAPI(UsdPhysics.CollisionAPI):
                    stage.RemovePrim(child.GetPath())
            for index, collision in enumerate(body["collisions"]):
                hull = trimesh.load(out / collision["file"], force="mesh", process=False)
                node = mesh_prim(stage, path + f"/Collision_{index:03d}", hull)
                UsdPhysics.CollisionAPI.Apply(node.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(node.GetPrim()).CreateApproximationAttr("convexHull")
                node.CreateVisibilityAttr("invisible")
                pose(node.GetPrim(), np.asarray(collision["origin"]))
        # Preserve the old no-self-collision policy, including the two added bodies.
        UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().SetTargets(
            [Sdf.Path("/Robot/" + b["name"]) for b in bodies if b["name"] != name])
    for item in gimbal["joints"]:
        node = UsdPhysics.RevoluteJoint.Define(stage, "/Robot/tree_joints/" + item["name"])
        node.CreateBody0Rel().SetTargets(["/Robot/" + item["parent"]])
        node.CreateBody1Rel().SetTargets(["/Robot/" + item["child"]])
        alignment = Rotation.align_vectors([item["axis"]], [[0, 0, 1]])[0].as_matrix()
        joint_frame = np.asarray(item["origin"])
        node.CreateAxisAttr("Z")
        node.CreateLocalPos0Attr(Gf.Vec3f(*joint_frame[:3, 3]))
        node.CreateLocalPos1Attr(Gf.Vec3f(0))
        node.CreateLocalRot0Attr(quat(joint_frame[:3, :3] @ alignment, single=True))
        node.CreateLocalRot1Attr(quat(alignment, single=True))
        node.CreateExcludeFromArticulationAttr(False)
        if item["type"] == "revolute":
            lower, upper = np.rad2deg(item["mechanical_position_limits_rad"])
            node.CreateLowerLimitAttr(float(lower))
            node.CreateUpperLimitAttr(float(upper))
    stage.GetDefaultPrim().SetCustomData(dict(model_kind="full_vehicle_closedchain_candidate",
                                             hardware_deployment_ready=False))
    visuals.GetRootLayer().Save()
    stage.GetRootLayer().Save()


def static_validation(out: Path, core: dict, parent_core: dict, gimbal: dict) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    root = ET.parse(out / "robot.urdf").getroot()
    assert root.get("version") == URDF_FORMAT_VERSION
    assert core["joints"] == parent_core["joints"]
    assert core["constraints"] == parent_core["constraints"]
    assert core["nominal_joint_pos"] == parent_core["nominal_joint_pos"]
    poses = fk(core["joints"], core["nominal_joint_pos"])
    closure = []
    for constraint in core["constraints"]:
        p0 = poses[constraint["body0"]] @ [*constraint["local_pos0_m"], 1]
        p1 = poses[constraint["body1"]] @ [*constraint["local_pos1_m"], 1]
        closure.append(float(np.linalg.norm(p0[:3] - p1[:3])))
    stage = Usd.Stage.Open(str(out / "robot.usda"))
    rigid = [p.GetName() for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)]
    loops = [p.GetName() for p in joints if UsdPhysics.Joint(p).GetExcludeFromArticulationAttr().Get()]
    assert len(root.findall("link")) == len(rigid) == 21
    assert len(root.findall("joint")) == len(joints) - len(loops) == 20
    assert len(loops) == 6
    visual_readback = validate_renderable_visuals(stage, [body["name"] for body in core["bodies"] + gimbal["bodies"]])
    assert len(visual_readback) == 21
    for mesh in root.findall(".//mesh"):
        assert (out / mesh.get("filename")).is_file()
    urdf_joints = xml_joints(root)
    urdf_zero = fk(urdf_joints)
    expected_zero = fk(core["joints"] + gimbal["joints"])
    pose_error = max(float(np.abs(urdf_zero[n] - expected_zero[n]).max()) for n in expected_zero)
    assert pose_error < 1e-12
    cache = UsdGeom.XformCache()
    auxiliary_zero_frames = {}
    for name in gimbal["joint_names"]:
        joint = UsdPhysics.Joint(stage.GetPrimAtPath("/Robot/tree_joints/" + name))
        frames = []
        for side in (0, 1):
            target = getattr(joint, f"GetBody{side}Rel")().GetTargets()[0]
            world = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(target))).T
            quaternion = getattr(joint, f"GetLocalRot{side}Attr")().Get()
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(
                [*quaternion.GetImaginary(), quaternion.GetReal()]).as_matrix()
            local[:3, 3] = getattr(joint, f"GetLocalPos{side}Attr")().Get()
            frames.append(world @ local)
        position_error = float(np.linalg.norm(frames[0][:3, 3] - frames[1][:3, 3]))
        angle_error = float(Rotation.from_matrix(
            frames[0][:3, :3].T @ frames[1][:3, :3]).magnitude())
        assert position_error < 1e-6 and angle_error < 1e-6
        auxiliary_zero_frames[name] = dict(anchor_error_m=position_error,
                                           orientation_error_rad=angle_error)
    pitch = next(item for item in gimbal["joints"] if item["name"] == "M_joint2")
    pitch_xml = root.find("joint[@name='M_joint2']")
    assert pitch_xml.get("type") == pitch["type"] == "revolute"
    xml_bounds = [float(pitch_xml.find("limit").get(key)) for key in ("lower", "upper")]
    assert np.allclose(xml_bounds, np.deg2rad(PITCH_LIMITS_DEG), rtol=0, atol=1e-12)
    pitch_usd = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath("/Robot/tree_joints/M_joint2"))
    assert np.allclose([pitch_usd.GetLowerLimitAttr().Get(), pitch_usd.GetUpperLimitAttr().Get()],
                       PITCH_LIMITS_DEG, rtol=0, atol=1e-6)
    pitch_direction = pitch_elevation_check(gimbal)
    bodies = core["bodies"] + gimbal["bodies"]
    return dict(urdf_bodies=len(rigid), urdf_tree_joints=len(root.findall("joint")),
                usd_bodies=len(rigid), usd_tree_joints=len(joints) - len(loops),
                usd_external_loop_constraints=loops, core_joint_contract_unchanged=True,
                core_nominal_joint_pos_unchanged=True, core_constraint_anchors_unchanged=True,
                nominal_core_closure_max_m=max(closure), urdf_model_zero_pose_max_abs_error=pose_error,
                auxiliary_usd_zero_frame_errors=auxiliary_zero_frames,
                urdf_format_version=URDF_FORMAT_VERSION,
                renderable_visual_meshes=visual_readback, pitch_limits_deg=list(PITCH_LIMITS_DEG),
                pitch_limit_evidence="user_specified_not_measured", pitch_elevation_fk=pitch_direction,
                core_mechanical_limit_evidence="historical_parent_pending_new_CAD_travel_review",
                total_mass_kg=sum(b["mass"] for b in bodies),
                core_mass_kg=sum(b["mass"] for b in core["bodies"]),
                auxiliary_mass_kg=sum(b["mass"] for b in gimbal["bodies"]),
                physics_executed=False, collision_qualified=False)


def build(source: Path, parent: Path, out: Path, settings: dict) -> dict:
    manifest = json.loads((parent / "manifest.json").read_text())
    if "knee_inner_limits_deg" not in manifest:
        raise ValueError("Historical parent must state its knee limits; they are not newly qualified")
    for relative, digest in manifest["files_sha256"].items():
        if sha(parent / relative) != digest:
            raise ValueError(f"Parent asset hash mismatch: {relative}")
    parent_core = json.loads((parent / "model_spec.json").read_text())
    core = copy.deepcopy(parent_core)
    source_root = ET.parse(source).getroot()
    source_joints = xml_joints(source_root)
    new_frames, old_frames = fk(source_joints), fk(core["joints"])
    source_links = {n.get("name"): n for n in source_root.findall("link")}
    if set(source_links) != set(manifest["rigid_body_names"] + AUX_BODIES):
        raise ValueError("Unexpected full-vehicle body names")
    shutil.copytree(parent, out)
    (out / "manifest.json").unlink()
    shutil.copyfile(parent / "manifest.json", out / "core_parent_manifest.json")
    shutil.copyfile(source, out / "source_full_vehicle.urdf")
    mapping = {n: np.linalg.inv(old_frames[n]) @ ROOT_ROTATION @ new_frames[n] for n in old_frames}
    mapping.update({n: np.eye(4) for n in AUX_BODIES})
    new_bodies = []
    inertia_evidence, collision_evidence, source_pins = {}, {}, {str(source): sha(source)}
    for name in manifest["rigid_body_names"] + AUX_BODIES:
        link = source_links[name]
        mass, com, tensor = inertia(link, mapping[name])
        source_mesh = source.parent.parent / "meshes" / Path(link.find("visual/geometry/mesh").get("filename")).name
        target = out / "meshes" / (name + ".stl")
        shutil.copyfile(source_mesh, target)
        source_pins[str(source_mesh)] = sha(source_mesh)
        body = next((b for b in core["bodies"] if b["name"] == name), None)
        if body is None:
            body = dict(name=name, component=name, color=[.35, .38, .43])
            new_bodies.append(body)
        body.update(mass=mass, com=com.tolist(), inertia=tensor.tolist(),
                    mesh=f"meshes/{name}.stl",
                    mesh_origin=(mapping[name] @ origin(link.find("visual/origin"))).tolist())
        inertia_evidence[name] = dict(method="source_CAD_COM_inertia_rigid_frame_transport",
                                     source_to_bundle_body=mapping[name].tolist(),
                                     mass_kg=mass, com_m=com.tolist(), inertia_kg_m2=tensor.tolist())
        if name in ("base_link", *AUX_BODIES):
            print(f"Preparing collision candidates: {name}", flush=True)
            mesh = trimesh.load(source_mesh, force="mesh", process=True)
            body["collisions"], collision_evidence[name] = make_collisions(
                mesh, name, np.asarray(body["mesh_origin"]), out, settings)
    auxiliary_joints = [copy.deepcopy(j) for j in source_joints if j["name"] in AUX_JOINTS]
    auxiliary_joints.sort(key=lambda j: AUX_JOINTS.index(j["name"]))
    auxiliary_joints[0]["origin"] = (ROOT_ROTATION @ auxiliary_joints[0]["origin"]).tolist()
    for joint in auxiliary_joints:
        joint["effort_limit_nm"] = 4.5
        joint["effort_limit_source"] = "RMCS gimbal motor limit, user repository controller"
        joint["velocity_limit_rad_s"] = None
        joint["mechanical_position_limits_rad"] = None
    view_forward = fk(auxiliary_joints)["M_link2"][:3, :3].T @ [1., 0., 0.]
    gimbal = dict(schema_version=1, joint_names=AUX_JOINTS, body_names=AUX_BODIES,
                  joints=auxiliary_joints, bodies=new_bodies, view_body="M_link2",
                  view_forward_local=view_forward.tolist(), control_joint_signs=[1, -1],
                  mechanical_position_limits_status="unknown_not_inferred_from_SCUT",
                  control_target_limits_scope="runtime_config_not_mechanical_qualification")
    gimbal = configure_gimbal_pitch(gimbal)
    save(out / "model_spec.json", core)
    save(out / "gimbal_model.json", gimbal)
    save(out / "full_vehicle_coordinate_mapping.json", dict(
        schema_version=1, source_root_to_control=ROOT_ROTATION.tolist(),
        equation="p_bundle=inv(T_bundle_body_q0)*Rz(+pi/2)*T_source_body_q0*p_source_local",
        body_frames={n: m.tolist() for n, m in mapping.items()},
        core_frame_contract="unchanged_from_calibrated_parent", auxiliary_frames="source_local_frames"))
    save(out / "full_vehicle_inertial_sources.json", inertia_evidence)
    save(out / "full_vehicle_collision_proxies.json", collision_evidence)
    save(out / "full_vehicle_source_provenance.json", dict(
        source_urdf=str(source), source_files_sha256=source_pins,
        parent_asset=str(parent), parent_manifest_sha256=sha(parent / "manifest.json"),
        parent_asset_revision=manifest["asset_revision"],
        candidate_scope="new CAD masses/inertias/visuals/base collision/gimbal with calibrated core geometry",
        preserved_core_collision_scope="all wheel-leg links except base_link",
        full_vehicle_mass_confirmed_by_user=True))
    all_bodies = core["bodies"] + new_bodies
    write_urdf(parent, out, all_bodies, gimbal)
    write_usd(parent, out, all_bodies, gimbal)
    result = static_validation(out, core, parent_core, gimbal)
    save(out / "full_vehicle_static_validation.json", result)
    (out / "FULL_VEHICLE_CANDIDATE.md").write_text(
        "# Full-vehicle candidate\n\n"
        "Core coordinates, tree joints, six spherical loops and spring curves come from the pinned "
        "historical parent. Its knee limits and gas-spring travel remain historical inputs pending "
        "new CAD travel review; they are not newly confirmed mechanical ranges. "
        "Source CAD masses, COM and inertia are transported into the core frames. "
        "The model_spec.json remains the q18/19-body reset model; gimbal_model.json adds two bodies "
        "with continuous yaw and bounded revolute pitch. Total mass includes both.\n\n"
        "Core wheel/leg collision proxies are preserved. Base and gimbal proxies are CPU CoACD "
        "candidates, not qualified contact geometry. New source visual meshes are retained without "
        "ownership cleanup; inherited mesh_ownership/inertial_sources documents describe only the "
        "parent asset. Full-vehicle evidence is in full_vehicle_* files.\n\n"
        "Gimbal effort limit is 4.5 Nm from RMCS. Native pitch is bounded to -30/+35 degrees "
        "relative to source zero by the user's instruction, not measured-stop evidence. "
        "Yaw stops and speed limits remain unknown; runtime supplies the effort controller. "
        "The URDF declares version 1.2 and requires a compatible parser: omitted velocity "
        "uses the format's infinity default, not a measured unlimited hardware speed. "
        "Isaac training loads the USD directly and is unaffected by URDF parser compatibility. "
        "Specification: https://github.com/ros/urdfdom/blob/rolling/README.md . "
        "No physical simulation or training was run by this converter.\n"
    )
    shutil.copyfile(Path(__file__), out / "tools" / Path(__file__).name)
    manifest.update(asset_revision=REVISION, parent_manifest_sha256=sha(parent / "manifest.json"),
                    urdf_format_version=URDF_FORMAT_VERSION,
                    urdf_parser_requirement="URDF 1.2 compatible parser",
                    urdf_omitted_velocity_semantics="format default infinity; hardware speed unknown, not measured unlimited",
                    native_training_asset_format="USD",
                    source_urdf="source_full_vehicle.urdf", source_urdf_sha256=sha(source),
                    auxiliary_joint_names=AUX_JOINTS,
                    auxiliary_body_names=AUX_BODIES, auxiliary_model="gimbal_model.json",
                    nominal_joint_pos={**manifest["nominal_joint_pos"], **dict.fromkeys(AUX_JOINTS, 0.)},
                    total_mass_kg=result["total_mass_kg"], model_spec_scope="core_19body_18joint_reset_model",
                    core_mass_kg=result["core_mass_kg"], auxiliary_mass_kg=result["auxiliary_mass_kg"],
                    inertial_sources="full_vehicle_inertial_sources.json",
                    collision_proxy_evidence="full_vehicle_collision_proxies.json",
                    coordinate_mapping="full_vehicle_coordinate_mapping.json",
                    parent_evidence_scope="inherited mesh_ownership/inertial_sources are parent-only historical evidence",
                    mechanical_scope="historical core knee/spring limits pending new CAD travel review; user-specified pitch down30/up35",
                    core_mechanical_limit_evidence="historical_parent_not_newly_qualified",
                    spring_travel_evidence="historical_parent_pending_new_CAD_travel_review",
                    gimbal_pitch_limits_deg=list(PITCH_LIMITS_DEG),
                    gimbal_pitch_limit_source="user_specified_20261101_not_measured",
                    cpu_geometry_comparison_passed=True, static_validation_passed=True,
                    dynamics_validation="pending_full_vehicle", physx_validation="not_executed",
                    training_ready=False, physics_qualified=False, hardware_deployment_ready=False,
                    collision_proxy_status="clearance_carved_coacd_cpu_candidate",
                    source_mass_confirmation="user_confirmed_full_vehicle_and_chassis_mass_increase",
                    files_sha256={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*"))
                                  if p.is_file() and "__pycache__" not in p.parts})
    save(out / "manifest.json", manifest)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-urdf", required=True, type=Path)
    parser.add_argument("--core-asset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-convex-hulls", default=24, type=int)
    parser.add_argument("--collision-threshold", default=.03, type=float)
    args = parser.parse_args()
    source, parent, target = args.source_urdf.resolve(), args.core_asset.resolve(), args.output.resolve()
    if target.exists() or target.is_relative_to(parent) or target.is_relative_to(source.parent.parent):
        parser.error("Output must be a new directory outside the source and calibrated core")
    settings = dict(threshold=args.collision_threshold, max_convex_hull=args.max_convex_hulls,
                    preprocess_mode="on", preprocess_resolution=50, resolution=2000,
                    mcts_nodes=20, mcts_iterations=100, mcts_max_depth=3,
                    merge=True, decimate=True, max_ch_vertex=255, seed=0)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".full-vehicle-build-", dir=target.parent) as temporary:
        output = Path(temporary) / "asset"
        result = build(source, parent, output, settings)
        shutil.move(str(output), str(target))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
