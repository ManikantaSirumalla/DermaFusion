"""
DermaFusion Body Model Segmenter for Blender
=============================================

PURPOSE:
    Takes a realistic humanoid mesh (e.g. from MakeHuman) and segments it
    into 12 named objects matching the HAM10000 body regions, applies a
    skin material, then exports as COLLADA (.dae) for SceneKit.

USAGE:
    1. Open Blender
    2. Import your humanoid mesh (File > Import > whatever format)
    3. Select the body mesh object
    4. Open Scripting workspace (or Text Editor)
    5. Paste this script
    6. Click "Run Script" (or Alt+P)
    7. Exported file appears at the configured path

REQUIREMENTS:
    - Blender 3.6+ (tested on 4.0+)
    - A humanoid mesh in T-pose or A-pose, facing -Y (Blender default)
    - Mesh should be a single object (join parts first with Ctrl+J if needed)

WHAT IT DOES:
    Step 1: Analyzes mesh bounding box to determine body proportions
    Step 2: Assigns each face to one of 12 regions based on vertex positions
    Step 3: Separates into 12 named objects (region_scalp, region_face, etc.)
    Step 4: Applies a Phong skin material to all regions
    Step 5: Adds camera + 3-point lighting
    Step 6: Exports as .dae (COLLADA) for SceneKit

OUTPUT NODE NAMES (exact match required for iOS app):
    region_scalp, region_face, region_ear, region_neck,
    region_chest, region_abdomen, region_back,
    region_upper_extremity, region_lower_extremity,
    region_hand, region_foot, region_genital

AUTHOR: DermaFusion build tooling
LICENSE: CC0 (this script), CC0 (MakeHuman exports)
"""

import bpy
import bmesh
import os
import math
from mathutils import Vector


# ═══════════════════════════════════════════════
# CONFIGURATION — Adjust these if needed
# ═══════════════════════════════════════════════

# Export path — change to your preferred location
EXPORT_PATH = os.path.expanduser("~/Desktop/BodyModel.dae")

# If True, also exports .obj alongside .dae
ALSO_EXPORT_OBJ = True

# Skin color (RGB, 0-1 range) — warm flesh tone matching reference image
SKIN_COLOR = (0.83, 0.71, 0.62)

# Highlight color reference (not applied, just documented)
# The iOS app sets this at runtime: brandPrimary at 85% opacity
HIGHLIGHT_COLOR_REF = (0.27, 0.52, 0.96)

# Whether to add camera and lights to the scene
ADD_CAMERA_AND_LIGHTS = True


# ═══════════════════════════════════════════════
# REGION CLASSIFICATION
# ═══════════════════════════════════════════════

def classify_vertex_region(co, bounds):
    """
    Classify a vertex position into one of 12 HAM10000 body regions.

    Uses normalized coordinates relative to the mesh bounding box.
    The mesh is assumed to be in T-pose or A-pose, Y-up or Z-up
    (auto-detected from bounding box aspect ratio).

    Args:
        co: vertex world-space coordinate (Vector)
        bounds: dict with keys 'min', 'max', 'center', 'size',
                'up_axis', 'forward_axis', 'right_axis'

    Returns:
        str: region name (e.g. 'region_scalp')
    """
    # Normalize coordinates to 0-1 range within bounding box
    size = bounds['size']
    mn = bounds['min']

    # Determine which axis is "up" (tallest dimension = height)
    up = bounds['up_axis']       # index: 0=X, 1=Y, 2=Z
    fwd = bounds['forward_axis'] # front-back axis
    right = bounds['right_axis'] # left-right axis

    # Normalized position (0 = bottom/left/back, 1 = top/right/front)
    h = (co[up] - mn[up]) / size[up] if size[up] > 0 else 0.5         # height: 0=feet, 1=head
    lr = (co[right] - mn[right]) / size[right] if size[right] > 0 else 0.5  # left-right: 0.5=center
    fb = (co[fwd] - mn[fwd]) / size[fwd] if size[fwd] > 0 else 0.5   # front-back

    # Distance from center axis (for limb detection)
    center_lr = 0.5
    lr_dist = abs(lr - center_lr)

    # ── HEAD REGIONS (top 12% of height) ──
    if h > 0.88:
        # Ears: far left/right of head
        if lr_dist > 0.08:
            return 'region_ear'
        # Scalp: top of head (above ~93%)
        if h > 0.93:
            return 'region_scalp'
        # Face: front half of remaining head
        if fb > 0.45:
            return 'region_face'
        # Back of head → scalp
        return 'region_scalp'

    # ── NECK (88% to 82% of height) ──
    if h > 0.82 and lr_dist < 0.10:
        return 'region_neck'

    # ── ARMS & HANDS (far from center, above hip line) ──
    if lr_dist > 0.18 and h > 0.45:
        # Hands: lowest part of arm reach (narrow band at wrist/hand level)
        if h < 0.55 or lr_dist > 0.42:
            return 'region_hand'
        return 'region_upper_extremity'

    # ── TORSO (82% to 50% of height, near center) ──
    if h > 0.50 and lr_dist < 0.18:
        # Genital area: very specific small zone at bottom-front of torso
        if h < 0.54 and fb > 0.45 and lr_dist < 0.06:
            return 'region_genital'

        # Back vs front: split at midline
        if fb < 0.40:
            return 'region_back'

        # Upper torso (chest) vs lower torso (abdomen)
        if h > 0.65:
            return 'region_chest'
        else:
            return 'region_abdomen'

    # Shoulder/upper arm transition
    if h > 0.72 and lr_dist >= 0.18:
        return 'region_upper_extremity'

    # ── LEGS & FEET (below 50% of height) ──
    if h <= 0.50:
        # Feet: below ~5%
        if h < 0.05:
            return 'region_foot'
        return 'region_lower_extremity'

    # ── FALLBACK ──
    # Anything remaining that's near center at mid-torso
    if fb < 0.40:
        return 'region_back'
    if h > 0.65:
        return 'region_chest'
    return 'region_abdomen'


def analyze_mesh_bounds(obj):
    """
    Analyze mesh bounding box and determine axis orientation.

    Returns:
        dict with 'min', 'max', 'center', 'size', and axis indices
    """
    # Get world-space bounding box
    bbox_world = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]

    min_co = Vector((
        min(v.x for v in bbox_world),
        min(v.y for v in bbox_world),
        min(v.z for v in bbox_world),
    ))
    max_co = Vector((
        max(v.x for v in bbox_world),
        max(v.y for v in bbox_world),
        max(v.z for v in bbox_world),
    ))

    size = max_co - min_co
    center = (min_co + max_co) / 2

    # Determine up axis (tallest dimension = height of human)
    dims = [size.x, size.y, size.z]
    up_axis = dims.index(max(dims))

    # Determine right axis (widest after height = arm span or shoulder width)
    remaining = [(dims[i], i) for i in range(3) if i != up_axis]
    remaining.sort(reverse=True)
    right_axis = remaining[0][1]  # wider = left-right (T-pose arm span)
    forward_axis = remaining[1][1]  # narrower = front-back depth

    print(f"  Bounding box: {size.x:.3f} x {size.y:.3f} x {size.z:.3f}")
    print(f"  Up axis: {'XYZ'[up_axis]}, Right axis: {'XYZ'[right_axis]}, Forward axis: {'XYZ'[forward_axis]}")
    print(f"  Height: {size[up_axis]:.3f}, Width: {size[right_axis]:.3f}, Depth: {size[forward_axis]:.3f}")

    return {
        'min': min_co,
        'max': max_co,
        'center': center,
        'size': size,
        'up_axis': up_axis,
        'forward_axis': forward_axis,
        'right_axis': right_axis,
    }


# ═══════════════════════════════════════════════
# MESH SEGMENTATION
# ═══════════════════════════════════════════════

def segment_mesh(obj):
    """
    Segment a humanoid mesh into 12 body region objects.

    Args:
        obj: The Blender mesh object to segment

    Returns:
        list: Names of created region objects
    """
    print(f"\n{'='*60}")
    print(f"DermaFusion Body Segmenter")
    print(f"{'='*60}")
    print(f"Input mesh: '{obj.name}' ({len(obj.data.polygons)} faces, {len(obj.data.vertices)} verts)")

    # Ensure we're in object mode
    bpy.ops.object.mode_set(mode='OBJECT')

    # Analyze bounds
    bounds = analyze_mesh_bounds(obj)

    # Apply transforms so world coords match local coords
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Classify each face by majority vertex region
    mesh = obj.data
    face_regions = {}

    region_counts = {}
    for poly in mesh.polygons:
        # Get world-space centroid of face
        face_center = Vector((0, 0, 0))
        for vi in poly.vertices:
            face_center += mesh.vertices[vi].co
        face_center /= len(poly.vertices)

        region = classify_vertex_region(face_center, bounds)
        face_regions[poly.index] = region

        region_counts[region] = region_counts.get(region, 0) + 1

    print(f"\nFace classification:")
    for name in sorted(region_counts.keys()):
        print(f"  {name:35s}  {region_counts[name]:6d} faces")

    # Separate into individual objects by region
    all_region_names = [
        'region_scalp', 'region_face', 'region_ear', 'region_neck',
        'region_chest', 'region_abdomen', 'region_back',
        'region_upper_extremity', 'region_lower_extremity',
        'region_hand', 'region_foot', 'region_genital'
    ]

    created_objects = []

    for region_name in all_region_names:
        # Select faces belonging to this region
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.mode_set(mode='OBJECT')

        for poly in mesh.polygons:
            poly.select = (face_regions.get(poly.index) == region_name)

        # Separate selected faces into new object
        bpy.ops.object.mode_set(mode='EDIT')
        try:
            bpy.ops.mesh.separate(type='SELECTED')
        except RuntimeError:
            # No faces selected for this region — skip
            print(f"  WARNING: No faces assigned to {region_name}")
            bpy.ops.object.mode_set(mode='OBJECT')
            continue
        bpy.ops.object.mode_set(mode='OBJECT')

        # The newly separated object is the last in selection
        # Find it — it will have a name like "OriginalName.001"
        new_obj = None
        for o in bpy.context.selected_objects:
            if o != obj and o.name not in [r for r in created_objects]:
                new_obj = o
                break

        if new_obj is None:
            # Fallback: find most recently created mesh object
            for o in sorted(bpy.data.objects, key=lambda x: x.name, reverse=True):
                if o.type == 'MESH' and o != obj and o.name not in created_objects:
                    new_obj = o
                    break

        if new_obj:
            new_obj.name = region_name
            new_obj.data.name = f"{region_name}_mesh"
            created_objects.append(region_name)
            face_count = len(new_obj.data.polygons)
            vert_count = len(new_obj.data.vertices)
            print(f"  Created: {region_name} ({vert_count} verts, {face_count} faces)")

        # Re-select original for next iteration
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    # Delete the remaining original object (should have no faces left)
    remaining_faces = len(obj.data.polygons)
    if remaining_faces > 0:
        print(f"\n  NOTE: {remaining_faces} faces remain unassigned in original mesh")
        # Rename remaining as back (most likely unclassified torso back faces)
        obj.name = "region_unassigned"
    else:
        bpy.data.objects.remove(obj, do_unlink=True)
        print(f"\n  Original mesh removed (all faces assigned)")

    print(f"\nCreated {len(created_objects)} region objects: {', '.join(created_objects)}")
    return created_objects


# ═══════════════════════════════════════════════
# MATERIAL APPLICATION
# ═══════════════════════════════════════════════

def apply_skin_material():
    """
    Ensure all region objects have a proper skin material.

    If the original mesh had an MPFB2/MakeHuman skin material (with textures,
    normal maps, etc.), each separated region PRESERVES that material
    automatically — no action needed.

    If regions have no material (e.g. mesh had none), creates a fallback
    Principled BSDF skin material matching the reference image.

    The iOS app dynamically changes material.diffuse.contents at runtime
    for highlighting, so we just need a baseline material here.
    """
    print(f"\nChecking skin materials...")

    region_objects = [o for o in bpy.data.objects if o.name.startswith('region_') and o.type == 'MESH']
    
    # Check if regions already have materials (from MPFB2 skin)
    regions_with_material = sum(1 for o in region_objects if len(o.data.materials) > 0)
    
    if regions_with_material == len(region_objects):
        print(f"  ✓ All {len(region_objects)} regions already have materials (preserved from source mesh)")
        print(f"    Skin textures, normal maps, and UV coordinates are intact.")
        
        # Just ensure smooth shading on all
        for obj in region_objects:
            for poly in obj.data.polygons:
                poly.use_smooth = True
        
        # Tweak for matte medical look: reduce specular if Principled BSDF
        for obj in region_objects:
            for mat in obj.data.materials:
                if mat and mat.use_nodes:
                    for node in mat.node_tree.nodes:
                        if node.type == 'BSDF_PRINCIPLED':
                            # Make slightly more matte for medical app look
                            try:
                                node.inputs['Roughness'].default_value = max(
                                    node.inputs['Roughness'].default_value, 0.75
                                )
                            except (KeyError, IndexError):
                                pass
                            for spec_name in ['Specular IOR Level', 'Specular']:
                                try:
                                    node.inputs[spec_name].default_value = min(
                                        node.inputs[spec_name].default_value, 0.2
                                    )
                                    break
                                except (KeyError, IndexError):
                                    continue
        print(f"    Adjusted roughness/specular for matte medical appearance.")
        return

    # Fallback: create a skin material for regions that lack one
    print(f"  {regions_with_material}/{len(region_objects)} regions have materials")
    print(f"  Creating fallback skin material for the rest...")

    mat = bpy.data.materials.new(name="DermaFusion_Skin")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    for node in nodes:
        nodes.remove(node)

    # Principled BSDF — matte skin matching reference image
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (0, 0)
    bsdf.inputs['Base Color'].default_value = (*SKIN_COLOR, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.85      # Very matte (reference is matte)
    # Handle Blender version differences for Subsurface input name
    for ss_name in ['Subsurface Weight', 'Subsurface']:
        try:
            bsdf.inputs[ss_name].default_value = 0.01  # Subtle skin translucency
            break
        except KeyError:
            continue
    try:
        bsdf.inputs['Specular IOR Level'].default_value = 0.15  # Minimal shine
    except KeyError:
        try:
            bsdf.inputs['Specular'].default_value = 0.15  # Blender < 4.0
        except KeyError:
            pass

    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

    for obj in region_objects:
        if len(obj.data.materials) == 0:
            obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.use_smooth = True

    print(f"  Applied fallback skin to regions without materials")
    print(f"  Skin color: RGB({SKIN_COLOR[0]:.2f}, {SKIN_COLOR[1]:.2f}, {SKIN_COLOR[2]:.2f})")
    print(f"  For best results, load an MPFB2 skin BEFORE running this script.")


# ═══════════════════════════════════════════════
# SCENE SETUP (Camera + Lights)
# ═══════════════════════════════════════════════

def setup_scene():
    """Add camera and 3-point lighting for the body model."""
    if not ADD_CAMERA_AND_LIGHTS:
        return

    print(f"\nSetting up scene...")

    # Find the combined bounding box of all regions
    all_region_objs = [o for o in bpy.data.objects if o.name.startswith('region_')]
    if not all_region_objs:
        return

    all_coords = []
    for obj in all_region_objs:
        for v in obj.data.vertices:
            all_coords.append(obj.matrix_world @ v.co)

    min_y = min(c[1] for c in all_coords) if all_coords else 0
    max_y = max(c[1] for c in all_coords) if all_coords else 1.8
    center_y = (min_y + max_y) / 2

    # Determine up axis from data
    min_z = min(c[2] for c in all_coords)
    max_z = max(c[2] for c in all_coords)
    height_y = max_y - min_y
    height_z = max_z - min_z

    if height_z > height_y:
        # Z-up model
        cam_height = (min_z + max_z) / 2
        cam_pos = (0, -3.0, cam_height)
        cam_rot = (math.radians(90), 0, 0)
    else:
        # Y-up model
        cam_height = center_y
        cam_pos = (0, cam_height, 3.0)
        cam_rot = (math.radians(-5), 0, 0)

    # Camera
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50  # 50mm focal length
    cam_data.clip_start = 0.01
    cam_data.clip_end = 100
    cam_obj = bpy.data.objects.new("CameraNode", cam_data)
    cam_obj.location = cam_pos
    cam_obj.rotation_euler = cam_rot
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj
    print(f"  Camera at {cam_pos}")

    # Key light (upper right front)
    key_data = bpy.data.lights.new("DirectionalLight", 'SUN')
    key_data.energy = 2.5
    key_data.color = (1.0, 0.98, 0.95)
    key_obj = bpy.data.objects.new("DirectionalLightNode", key_data)
    key_obj.location = (2, 3, 2)
    key_obj.rotation_euler = (math.radians(-35), math.radians(30), 0)
    bpy.context.scene.collection.objects.link(key_obj)
    print(f"  Key light (Sun) at (2, 3, 2)")

    # Fill light (left)
    fill_data = bpy.data.lights.new("FillLight", 'SUN')
    fill_data.energy = 0.8
    fill_data.color = (0.9, 0.92, 1.0)
    fill_obj = bpy.data.objects.new("FillLightNode", fill_data)
    fill_obj.location = (-1.5, 2, 1)
    fill_obj.rotation_euler = (math.radians(-25), math.radians(-40), 0)
    bpy.context.scene.collection.objects.link(fill_obj)

    # Ambient / world
    world = bpy.data.worlds.new("DermaFusion_World")
    world.use_nodes = True
    bg = world.node_tree.nodes.get('Background')
    if bg:
        bg.inputs['Color'].default_value = (0.15, 0.15, 0.16, 1.0)
        bg.inputs['Strength'].default_value = 0.5
    bpy.context.scene.world = world

    print(f"  3-point lighting configured")


# ═══════════════════════════════════════════════
# EXPORT
# ═══════════════════════════════════════════════

def export_model():
    """Export the segmented model as COLLADA (.dae), OBJ, and glTF."""
    print(f"\nExporting...")

    # Select all region objects + camera + lights
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.data.objects:
        if obj.name.startswith('region_') or obj.name in ('CameraNode', 'DirectionalLightNode', 'FillLightNode'):
            obj.select_set(True)

    # Export COLLADA (.dae)
    dae_path = EXPORT_PATH
    try:
        bpy.ops.wm.collada_export(
            filepath=dae_path,
            selected=True,
            apply_modifiers=True,
            triangulate=True,
            include_material_textures=True,
        )
        file_size = os.path.getsize(dae_path) / 1024
        print(f"  ✓ COLLADA: {dae_path} ({file_size:.0f} KB)")
    except Exception as e:
        print(f"  ✗ COLLADA export failed: {e}")

    # Export OBJ (Blender 4.x uses wm.obj_export, older uses export_scene.obj)
    if ALSO_EXPORT_OBJ:
        obj_path = dae_path.replace('.dae', '.obj')
        try:
            # Blender 4.x API
            bpy.ops.wm.obj_export(
                filepath=obj_path,
                export_selected_objects=True,
                export_triangulated_mesh=True,
                export_materials=True,
                export_normals=True,
                export_uv=True,
            )
            file_size = os.path.getsize(obj_path) / 1024
            print(f"  ✓ OBJ: {obj_path} ({file_size:.0f} KB)")
        except AttributeError:
            try:
                # Blender 3.x fallback
                bpy.ops.export_scene.obj(
                    filepath=obj_path,
                    use_selection=True,
                    use_mesh_modifiers=True,
                    use_triangles=True,
                    use_materials=True,
                    use_normals=True,
                    use_uvs=True,
                )
                file_size = os.path.getsize(obj_path) / 1024
                print(f"  ✓ OBJ: {obj_path} ({file_size:.0f} KB)")
            except Exception as e:
                print(f"  ✗ OBJ export failed: {e}")

    # Export glTF (best material/texture preservation for USDZ pipeline)
    glb_path = dae_path.replace('.dae', '.glb')
    try:
        bpy.ops.export_scene.gltf(
            filepath=glb_path,
            export_format='GLB',
            use_selection=True,
            export_apply=True,
            export_materials='EXPORT',
            export_image_format='AUTO',
        )
        file_size = os.path.getsize(glb_path) / 1024
        print(f"  ✓ glTF: {glb_path} ({file_size:.0f} KB)")
        print(f"    → Convert .glb to .usdz using Apple Reality Converter")
        print(f"    → Then in Xcode: Editor > Convert to SceneKit (.scn)")
    except Exception as e:
        print(f"  ✗ glTF export failed: {e}")
        print(f"    (Enable glTF add-on: Edit > Preferences > Add-ons > glTF)")


# ═══════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════

def verify_output():
    """Verify all 12 region objects exist with correct names."""
    required = [
        'region_scalp', 'region_face', 'region_ear', 'region_neck',
        'region_chest', 'region_abdomen', 'region_back',
        'region_upper_extremity', 'region_lower_extremity',
        'region_hand', 'region_foot', 'region_genital'
    ]

    print(f"\n{'='*60}")
    print(f"VERIFICATION")
    print(f"{'='*60}")

    found = []
    missing = []

    for name in required:
        obj = bpy.data.objects.get(name)
        if obj and obj.type == 'MESH':
            verts = len(obj.data.vertices)
            faces = len(obj.data.polygons)
            has_mat = len(obj.data.materials) > 0
            has_uv = len(obj.data.uv_layers) > 0
            print(f"  ✓ {name:35s}  {verts:5d} verts  {faces:5d} faces  mat:{'✓' if has_mat else '✗'}  uv:{'✓' if has_uv else '–'}")
            found.append(name)
        else:
            print(f"  ✗ {name:35s}  MISSING")
            missing.append(name)

    total_verts = sum(len(bpy.data.objects[n].data.vertices) for n in found)
    total_faces = sum(len(bpy.data.objects[n].data.polygons) for n in found)

    print(f"\n  TOTAL: {total_verts} vertices, {total_faces} faces")
    print(f"  Found: {len(found)}/12 regions")

    if missing:
        print(f"\n  ⚠ MISSING REGIONS: {', '.join(missing)}")
        print(f"    This may happen if the mesh doesn't have geometry in that area.")
        print(f"    Adjust the classify_vertex_region() thresholds if needed.")
    else:
        print(f"\n  ✓ ALL 12 REGIONS PRESENT — Ready for SceneKit")

    return len(missing) == 0


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    """
    Main entry point. Expects the active object to be the humanoid mesh.
    """
    obj = bpy.context.active_object

    if obj is None or obj.type != 'MESH':
        print("ERROR: Please select a mesh object first!")
        print("  1. Click on your humanoid mesh in the viewport")
        print("  2. Run this script again")
        return

    # Clean up any previous DermaFusion exports
    for o in list(bpy.data.objects):
        if o.name.startswith('region_') or o.name in ('CameraNode', 'DirectionalLightNode', 'FillLightNode'):
            bpy.data.objects.remove(o, do_unlink=True)

    # Run the pipeline
    created = segment_mesh(obj)
    apply_skin_material()
    setup_scene()

    if verify_output():
        export_model()
        print(f"\n{'='*60}")
        print(f"DONE — Import into Xcode:")
        print(f"  1. Drag .dae into Resources/")
        print(f"  2. Editor > Convert to SceneKit Scene File (.scn)")
        print(f"  3. Or: Open .glb in Reality Converter > Export .usdz")
        print(f"{'='*60}")
    else:
        print(f"\nExport skipped — fix missing regions first.")
        print(f"You can manually adjust region boundaries in classify_vertex_region()")


# Run
main()
