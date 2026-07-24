import numpy as np
import trimesh
import fcl
from scipy.spatial import KDTree

import MeshCheck

def generate_tree_supports_with_tips(
    model_path, 
    contact_points, 
    output_path, 
    step_size=1.5, 
    merge_dist=4.0, 
    clearance=1.5, 
    tip_radius=0.5,      # Extremely thin for crisp, toolless snapping
    base_radius=4.0,     # Wide for maximum bed plate adhesion
    interface_gap=3.0    # Distance (mm) the tip stays thin before tapering starts
):
    # 1. Load the model mesh
    mesh = MeshCheck.load_mesh(model_path)
    contact_points = mesh.triangles_center[contact_points]
    floor_z = mesh.bounds[0][2]

    # 2. Build the FCL BVH Mesh structure
    fcl_mesh = fcl.BVHModel()
    fcl_mesh.beginModel(len(mesh.vertices), len(mesh.faces))
    fcl_mesh.addSubModel(mesh.vertices, mesh.faces)
    fcl_mesh.endModel()
    model_geom = fcl.CollisionObject(fcl_mesh, fcl.Transform())

    active_branches = {
        i: [np.array(pt)] for i, pt in enumerate(contact_points)
    }
    finished_paths = []
    branch_origins_z = {i: pt[2] for i, pt in enumerate(contact_points)}

    max_z = max([pt[2] for pt in contact_points])
    max_steps = int((max_z - floor_z) / step_size) + 5

    # 3. Layer-by-Layer Descent Loop
    for _ in range(max_steps):
        if not active_branches:
            break

        next_branches = {}
        for b_id, path in active_branches.items():
            current_tip = path[-1]

            if current_tip[2] <= floor_z + 0.1:
                finished_paths.append((b_id, path))
                continue

            next_z = max(current_tip[2] - step_size, floor_z)
            proposed_tip = np.array([current_tip[0], current_tip[1], next_z])

            # Dynamic radius calculation
            dist_from_origin = branch_origins_z[b_id] - proposed_tip[2]
            if dist_from_origin <= interface_gap:
                current_radius = tip_radius
            else:
                remaining_height = (
                    branch_origins_z[b_id] - floor_z - interface_gap
                )
                remaining_height = max(remaining_height, 1.0)
                taper_progress = (
                    dist_from_origin - interface_gap
                ) / remaining_height
                current_radius = tip_radius + (
                    base_radius - tip_radius
                ) * np.clip(taper_progress, 0.0, 1.0)

            # FCL collision avoidance
            sphere_radius = current_radius + clearance
            sphere_geom_type = fcl.Sphere(sphere_radius)
            sphere_transform = fcl.Transform(proposed_tip)
            sphere_obj = fcl.CollisionObject(
                sphere_geom_type, sphere_transform
            )

            request = fcl.CollisionRequest(
                enable_contact=True, num_max_contacts=1
            )
            result = fcl.CollisionResult()
            fcl.collide(sphere_obj, model_geom, request, result)

            if result.is_collision:
                contact = result.contacts[0]
                penetration_depth = contact.penetration_depth
                normal = contact.normal

                push_vector = normal * penetration_depth
                push_vector[2] = 0.0  # Keep movement horizontal

                norm = np.linalg.norm(push_vector)
                if norm > 0:
                    proposed_tip += (push_vector / norm) * (
                        penetration_depth + 0.1
                    )
                else:
                    # Fallback if normal was perfectly vertical: push outward radially
                    proposed_tip[0] += 0.5

            next_branches[b_id] = path + [proposed_tip]

        if not next_branches:
            break

        # KD-Tree Branch Merging Pipeline
        b_ids = list(next_branches.keys())
        tips = np.array([next_branches[b_id][-1] for b_id in b_ids])

        if len(tips) > 1:
            tree = KDTree(tips)
            pairs = tree.query_pairs(r=merge_dist)
            merged_ids = set()

            for i, j in pairs:
                id_i, id_j = b_ids[i], b_ids[j]
                if id_i in merged_ids or id_j in merged_ids:
                    continue

                midpoint = (tips[i] + tips[j]) / 2.0
                next_branches[id_i][-1] = midpoint
                next_branches[id_j][-1] = midpoint

                # Save the dead branch path up to the merging node
                finished_paths.append((id_j, next_branches[id_j]))
                merged_ids.add(id_j)

            active_branches = {
                b_id: path
                for b_id, path in next_branches.items()
                if b_id not in merged_ids
            }
        else:
            active_branches = next_branches

    # Collect any paths that stayed active until the loop finished
    for b_id, path in active_branches.items():
        finished_paths.append((b_id, path))

    # --- SOLID MESH GENERATION ---
    support_segments = []
    for orig_id, path in finished_paths:
        if len(path) < 2:
            continue
        for idx in range(len(path) - 1):
            p1, p2 = path[idx], path[idx + 1]
            segment_vector = p2 - p1
            length = np.linalg.norm(segment_vector)
            if length < 0.001:
                continue

            d1 = branch_origins_z[orig_id] - p1[2]
            d2 = branch_origins_z[orig_id] - p2[2]
            h_total = max(branch_origins_z[orig_id] - floor_z - interface_gap, 1.0)

            # Evaluate radii
            r_top = (
                tip_radius
                if d1 <= interface_gap
                else tip_radius
                + (base_radius - tip_radius)
                * np.clip((d1 - interface_gap) / h_total, 0.0, 1.0)
            )
            r_bottom = (
                tip_radius
                if d2 <= interface_gap
                else tip_radius
                + (base_radius - tip_radius)
                * np.clip((d2 - interface_gap) / h_total, 0.0, 1.0)
            )

            # Generate accurate cone segments using trimesh primitive controls
            cone_segment = trimesh.creation.cylinder(
                radius=r_bottom, height=length
            )

            # Uniform scale transformation over the Z height spectrum
            # Shifting vertices temporarily to 0-to-height bounds makes scaling easy
            cone_segment.vertices[:, 2] += length / 2.0
            factors = 1.0 - (
                (1.0 - (r_bottom/r_top)) * (cone_segment.vertices[:, 2] / length)
            )
            cone_segment.vertices[:, :2] *= factors[:, np.newaxis]
            cone_segment.vertices[:, 2] -= length / 2.0

            # Orient and position segment
            transform_matrix = trimesh.geometry.align_vectors(
                np.array([0, 0, 1]), segment_vector
            )
            cone_segment.apply_transform(transform_matrix)
            cone_segment.apply_translation(p1 + (segment_vector / 2.0))
            support_segments.append(cone_segment)

    if support_segments:
        final_support_mesh = trimesh.util.concatenate(support_segments)
        final_support_mesh.export(output_path)
        print(f"Successfully saved supports to: {output_path}")
    else:
        print("Error: No support geometry nodes could be compiled.")

if __name__ == "__main__":
    origins = MeshCheck.main()
    
    generate_tree_supports_with_tips(
        model_path="../Tentacle_Remesh.stl", 
        contact_points=origins, 
        output_path="C:/Users/monto/Downloads/support_demo.stl", 
        step_size=1.0, 
        tip_radius=0.4, 
        base_radius=4, 
        interface_gap=1.0 
    )