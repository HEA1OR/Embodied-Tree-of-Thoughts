# Real2Sim

The reconstruction pipeline builds on **[SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects)**, **[FoundationPose](https://github.com/NVlabs/FoundationPose)**, and **[DexSim2Real<sup>2</sup>](https://github.com/jiangtaoran/DexSim2Real2)**.
For user convenience, we also provide simplified reconstruction options that support custom scenes and assets.

## Installation

You need to deploy [SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects)
(**we recommend using** [https://github.com/facebookresearch/sam-3d-objects/pull/38](https://github.com/facebookresearch/sam-3d-objects/pull/38) **so that meshes can be exported as `.obj` directly**),
[FoundationPose](https://github.com/NVlabs/FoundationPose), and optionally [DexSim2Real<sup>2</sup>](https://github.com/jiangtaoran/DexSim2Real2>) under the `/r2s` directory.
We also recommend installing them in **separate environments** to avoid dependency conflicts.

1. After deployment, you need to move the `demo.py` file to:

```
/r2s/sam-3d-objects/notebook/demo.py
```

2. Place the `etot` folder (containing our example images and masks) into:

```
/r2s/sam-3d-objects/notebook/images
```

The final directory structure should look like:

```
r2s/
├── adjust_size.py
├── test.ipynb
├── sam-3d-objects/
└── foundationpose/
```

---

## 1. Initial textured mesh reconstruction

First, you need to extract masks from RGB images using SAM-3 or any other method.
Note the following format differences:

* **SAM-3D-Objects mask**:
  4-channel PNG. Pixels kept in the mask must have **alpha = 255**, masked pixels have **alpha = 0**.
* **FoundationPose mask**:
  Single-channel PNG. Masked-in pixels = **255**, masked-out pixels = **0**.

We provide conversion scripts inside `test.ipynb`, which you may further modify as needed.

After preparing the RGB images and masks, set `IMAGE_PATH` to the correct directory and run:

```bash
conda activate sam-3d-objects
cd sam-3d-objects/notebook
python demo.py
```

You will obtain a textured mesh (`.glb`) for each mask.

> We also provide a fast, model-free way of obtaining masks using the **SAM-3 Web Playground**
> [https://aidemos.meta.com/segment-anything/editor/segment-image](https://aidemos.meta.com/segment-anything/editor/segment-image).
> Upload an image, specify the object name, apply the *color fill* effect to the selected mask (set color to `rgb(255,255,255)`), export the PNG, then convert it into the SAM-3D-Objects mask format using the scripts in `test.ipynb`.
> Note that **SAM-3D-Objects still must be deployed locally** for mesh export.
> If you want direct mesh generation, web-based tools such as **[Hunyuan3D](https://3d.hunyuan.tencent.com)** can generate meshes easily.

---

## 2. Mesh Size Adjustment

**Mesh Size Adjustment – `r2s/adjust_size.py`**

This script automatically adjusts the scale of 3D meshes so that their physical dimensions match real-world measurements estimated from RGB-D images and instance masks.
It computes an estimated metric scale for each mesh and generates a scaled version along with metadata.

### Usage

**Basic command**

```bash
python adjust_size.py --dataset_path PATH_TO_YOUR_DATASET
```

Your dataset directory must contain:

```
dataset/
 ├─ mesh/         # input meshes (.glb/.gltf/.obj/.ply)
 ├─ mask/         # instance masks (.png)
 ├─ raw_depth/    # depth images (.png)
 └─ raw_rgb/      # optional RGB images
```

**Example**

```bash
python adjust_mesh_size.py \
    --dataset_path ./data/sample_scene \
    --output_dir ./data/sample_scene/mesh_scaled
```

### Output Files

After running, the output directory will contain:

```
mesh_scaled/
 ├─ obj_a.glb
 ├─ obj_b.glb
 ...
```

Each file is the original mesh scaled according to the estimated real-world factor.
Because of camera noise, lighting, and other factors, the scale may still have small deviations, so additional manual adjustment may be needed.

---

## 3. Pose Adjustment

You should follow the requirements of FoundationPose:
place the RGB images, masks, depth maps, and `mesh_scaled` into the corresponding FoundationPose directories and rename them accordingly.
After updating the configuration, run `run_demo.py` to obtain the pose of each mesh relative to the camera.

**Important note:**
If you use the original version of [SAM-3D-Objects](https://github.com/facebookresearch/sam-3d-objects), the exported meshes are in `.glb` format.
Calling FoundationPose's `run_demo.py` may raise an error because `.glb` stores the mesh inside a scene structure, causing FoundationPose to fail to load textures.

To fix this, modify `run_demo.py` by adding the following code immediately after:

```python
mesh = trimesh.load(args.mesh_file)
```

Add:

```python
if isinstance(mesh, trimesh.Scene):
    mesh = mesh.dump(concatenate=True)
else:
    mesh = mesh
```

If you use the improved version
[https://github.com/facebookresearch/sam-3d-objects/pull/38](https://github.com/facebookresearch/sam-3d-objects/pull/38),
meshes are exported as `.obj` and this fix is unnecessary.

---

Before importing into simulation, you must compute the **absolute pose** of each mesh in the simulator.
Since the coordinate axes of a real-world camera differ from those used in simulation, you need to apply a coordinate transformation and extrinsic calibration. **The pose adjustment process is detailed in `test.ipynb`.**

> In practice, due to camera inaccuracies, lighting conditions, and algorithmic noise, the pose estimated in the real world may differ from the pose required in simulation. We recommend adjusting each object’s pose after importing into the simulator.
> OmniGibson uses JSON files to store scene configurations. We provide a simple script in `simworld/env_build/edit_scene.py` that lets you adjust the pose of objects in the simulator to the correct position and press **`z`** to save the updated JSON file.
> Alternatively, you can edit the numeric values in the scene JSON directly.

---

## 4. Articulated Objects

For articulated objects in the scene, we provide a convenient option using **[DexSim2Real<sup>2</sup>](https://github.com/jiangtaoran/DexSim2Real2)**.
Of course, simpler alternatives are also possible—for example, manually modeling the object, importing it into Blender, splitting it into parts, and configuring the articulation joints manually.

---



## 5. Importing into OmniGibson

There are two important considerations when importing reconstructed assets into **OmniGibson**:

#### **(1) Meshes must be converted to USD format**

All meshes must be converted into **USD** before they can be imported into OmniGibson.
You can refer to the USD files under the `assets` directory as examples of the required structure and conventions.

In addition, certain physical properties—such as **mass**, **friction coefficients**, and **collision parameters**—must be manually configured.
A convenient workflow is to open the USD file in **IsaacSim**, where you can visually inspect the mesh and adjust these physical attributes directly.

#### **(2) Handling non-convex objects and collision issues**

By default, OmniGibson computes collision volumes using **convex hull approximation**, which may cause non-convex objects (e.g., a pen holder) to become “sealed” internally and lose their hollow structure.

To address this, we recommend using [COACD](https://github.com/SarahWeiii/CoACD) to pre-segment the mesh into multiple convex parts before conversion.
This ensures accurate collision geometry and prevents undesired occlusion or filling inside non-convex shapes.

---
> We thank Rui Fang for his assistance with the 3D reconstruction component of this work.
> For any questions, please contact: **[2604317843xwj@gmail.com](mailto:2604317843xwj@gmail.com)** or **[fr23@mails.tsinghua.edu.cn](mailto:fr23@mails.tsinghua.edu.cn)**.



