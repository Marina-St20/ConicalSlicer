import argparse
import numpy as np
import trimesh
import fcl
from scipy.spatial import KDTree

import MeshCheck

def generate(
        mesh=None, 
        origins=None, 
        output_path="../support_demo.stl", 
        step_size=1.0, 
        tip_radius=0.4, 
        base_radius=4, 
):
    if mesh is None or origins is None:
        return
    

def main():
    parser = argparse.ArgumentParser(description='Generate supports for a mesh.')
    parser.add_argument('mesh_path', help='Path to the mesh file to load.')
    parser.add_argument('filter_path', help='Path to filter mesh.')
    parser.add_argument('output_path', help='Path to output.')
    parser.add_argument('--center', type=str, default=None, help='Optional center point as "x,y,z" passed to FindSupportFaces.py.')
    parser.add_argument('--bradius', type=float, default=4, help='Support base radius. Defaults to 4.0.')
    parser.add_argument('--tradius', type=float, default=.4, help='Support tip radius. Defaults to 0.4.')
    parser.add_argument('--offset', type=float, default=.2, help='Offset between supports and original mesh.')
    args = parser.parse_args()

    mesh = MeshCheck.load_mesh(args.mesh_path)
    origins = MeshCheck.vertical_scan()
    
    generate(
        model_path=mesh, 
        origins=origins, 
        output_path=args.output_path, 
        step_size=1.0, 
        tip_radius=args.tradius, 
        base_radius=args.bradius, 
    )

if __name__ == "__main__":
    main()
