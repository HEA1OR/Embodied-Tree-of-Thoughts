#!/usr/bin/env python3

import os
import json
import numpy as np
import open3d as o3d
import cv2
import trimesh
from typing import Tuple, Dict
import argparse


class MeshSizeAdjuster:
    def __init__(self, dataset_path: str):
        """
        Initialize the mesh size adjuster with the dataset root path.
        Required directory structure inside dataset_path:
            mesh/       - input meshes
            mask/       - instance masks
            raw_depth/  - depth images
            raw_rgb/    - optional RGB images
        """
        self.dataset_path = os.path.abspath(dataset_path)
        self.mesh_dir = os.path.join(dataset_path, "mesh")
        self.mask_dir = os.path.join(dataset_path, "mask")
        self.depth_dir = os.path.join(dataset_path, "raw_depth")
        self.rgb_dir = os.path.join(dataset_path, "raw_rgb")
        
        # Camera intrinsic matrix
        self.K = np.array([
            [906.461181640625, 0, 635.8511962890625],
            [0, 905.659912109375, 350.6916809082031],
            [0, 0, 1]
        ])
        
        # Depth scale factor (depth PNG is in millimeters)
        self.depth_scale = 1000.0
        
        # Validate essential directories
        assert os.path.exists(self.mesh_dir), f"Mesh directory not found: {self.mesh_dir}"
        assert os.path.exists(self.mask_dir), f"Mask directory not found: {self.mask_dir}"
        assert os.path.exists(self.depth_dir), f"Depth directory not found: {self.depth_dir}"
    
    def load_depth_from_pointcloud(self, basename: str) -> np.ndarray:
        """
        Load depth image and compute a representative real-world depth value
        from the valid mask region.
        """
        depth_files = [f for f in os.listdir(self.depth_dir) if f.endswith('.png')]
        if len(depth_files) == 0:
            raise ValueError(f"No depth image found in {self.depth_dir}")
        depth_path = os.path.join(self.depth_dir, depth_files[0])
        
        mask_path = os.path.join(self.mask_dir, f"{basename}.png")
        
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        # Convert to meters
        depth = depth / self.depth_scale
        
        # Valid region: non-zero depth, within range, and inside mask
        valid_mask = (depth > 0.1) & (depth < 1.75) & (mask > 127)
        
        if valid_mask.sum() < 100:
            print(f"Warning: {basename} has too few valid depth points ({valid_mask.sum()})")
            return None
        
        avg_depth = np.median(depth[valid_mask])
        return avg_depth
    
    def estimate_mesh_depth_from_bbox(self, mesh_path: str) -> float:
        """
        Load mesh and estimate:
            - center depth (Z center)
            - maximum size of bounding box
        """
        if mesh_path.endswith('.glb') or mesh_path.endswith('.gltf'):
            mesh_tri = trimesh.load(mesh_path, force='mesh')
        else:
            mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
            vertices = np.asarray(mesh_o3d.vertices)
            triangles = np.asarray(mesh_o3d.triangles)
            mesh_tri = trimesh.Trimesh(vertices=vertices, faces=triangles)
        
        bbox = mesh_tri.bounds
        
        center_z = (bbox[0, 2] + bbox[1, 2]) / 2
        
        size = bbox[1] - bbox[0]
        max_size = np.max(size)
        
        return center_z, max_size
    
    def calculate_scale_from_depth(self, basename: str, mesh_path: str, 
                                   initial_scale: float = 1.0) -> Tuple[float, Dict]:
        """
        Estimate mesh scaling factor based on depth, mask bbox, and camera intrinsics.
        """
        real_depth = self.load_depth_from_pointcloud(basename)
        if real_depth is None:
            return None, {"error": "Failed to get real depth"}
        
        mesh_center_z, mesh_max_size = self.estimate_mesh_depth_from_bbox(mesh_path)
        
        # Load mask
        mask_path = os.path.join(self.mask_dir, f"{basename}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        
        mask_binary = (mask > 127).astype(np.uint8)
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None, {"error": "No contours found in mask"}
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        
        real_width = (w * real_depth) / fx
        real_height = (h * real_depth) / fy
        real_max_size = max(real_width, real_height)
        
        estimated_scale = real_max_size / mesh_max_size if mesh_max_size > 0 else 1.0
        
        info = {
            "real_depth_m": float(real_depth),
            "mesh_original_size_m": float(mesh_max_size),
            "mask_bbox_pixels": [int(w), int(h)],
            "estimated_real_size_m": float(real_max_size),
            "calculated_scale": float(estimated_scale),
            "real_width_m": float(real_width),
            "real_height_m": float(real_height)
        }
        
        return estimated_scale, info
    
    def save_scaled_mesh(self, input_path: str, output_path: str, scale: float):
        """
        Save scaled mesh using Open3D or Trimesh depending on file type.
        """
        if input_path.endswith('.glb') or input_path.endswith('.gltf'):
            mesh = trimesh.load(input_path, force='mesh')
            mesh.apply_scale(scale)
            mesh.export(output_path)
        else:
            mesh = o3d.io.read_triangle_mesh(input_path)
            mesh.scale(scale, center=mesh.get_center())
            o3d.io.write_triangle_mesh(output_path, mesh)
        
        print(f"  Saved scaled mesh to: {output_path}")
    
    def process_all(self, output_dir: str = None, save_scaled_meshes: bool = True,
                    visualize: bool = False, save_npz: bool = True):
        """
        Process all meshes in mesh_dir, compute scale, and save results.
        """
        if output_dir is None:
            output_dir = os.path.join(self.dataset_path, "mesh_scaled")
        
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for filename in sorted(os.listdir(self.mesh_dir)):
            if not (filename.endswith('.glb') or filename.endswith('.obj') or 
                   filename.endswith('.ply') or filename.endswith('.gltf')):
                continue
            
            basename = os.path.splitext(filename)[0]
            print(f"\n{'='*60}")
            print(f"Processing: {basename}")
            print(f"{'='*60}")
            
            mesh_path = os.path.join(self.mesh_dir, filename)
            
            scale, info = self.calculate_scale_from_depth(basename, mesh_path)
            
            if scale is None:
                print(f"  Failed: {info.get('error', 'Unknown error')}")
                results[basename] = {"success": False, "error": info.get('error')}
                continue
            
            print(f"  Calculated scale: {scale:.4f}")
            print(f"  Info:")
            for key, value in info.items():
                if isinstance(value, float):
                    print(f"     - {key}: {value:.4f}")
                else:
                    print(f"     - {key}: {value}")
            
            results[basename] = {
                "success": True,
                "scale": scale,
                **info
            }
            
            if save_scaled_meshes:
                output_path = os.path.join(output_dir, filename)
                self.save_scaled_mesh(mesh_path, output_path, scale)
            
            if save_npz:
                identity_pose = np.eye(4)
                npz_path = os.path.join(output_dir, f"{basename}_pose_scaled.npz")
                np.savez(npz_path, pose=identity_pose, scale=scale)
                print(f"  Saved NPZ to: {npz_path}")
        
        # Save all results
        results_path = os.path.join(output_dir, "scale_results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Results saved to: {results_path}")
        print(f"Scaled meshes saved to: {output_dir}")
        print(f"{'='*60}\n")
        
        # Summary statistics
        successful = sum(1 for r in results.values() if r.get('success', False))
        total = len(results)
        print(f"Summary: {successful}/{total} meshes processed successfully")
        
        if successful > 0:
            scales = [r['scale'] for r in results.values() if r.get('success', False)]
            print(f"Scale range: {min(scales):.4f} - {max(scales):.4f}")
            print(f"Mean scale: {np.mean(scales):.4f} ± {np.std(scales):.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Adjust mesh sizes to match RGB and depth data.')
    parser.add_argument('--dataset_path', type=str, 
                       default='/home/agilex/R2S2R_tsinghua_ws/gary/build_kinematic/data1',
                       help='Path to the dataset')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: dataset_path/mesh_scaled)')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save scaled mesh files')
    parser.add_argument('--no-npz', action='store_true',
                       help='Do not save NPZ files (pose and scale)')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize results using Open3D')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Mesh Size Adjuster")
    print("="*60)
    print(f"Dataset path: {args.dataset_path}")
    print("="*60 + "\n")
    
    adjuster = MeshSizeAdjuster(args.dataset_path)
    results = adjuster.process_all(
        output_dir=args.output_dir,
        save_scaled_meshes=not args.no_save,
        save_npz=not args.no_npz,
        visualize=args.visualize
    )
    
    return results


if __name__ == "__main__":
    main()
