#!/usr/bin/env python3
"""
Generate DermaFusion body model as COLLADA (.dae) file.

COLLADA is XML-based, loads natively in SceneKit, and Xcode converts
it to .scn with: Editor > Convert to SceneKit Scene File Format (.scn)

Each body region is a separate <node> with its own <geometry>, so
SceneKit creates individual SCNNode objects for hit-testing.
"""

import math
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

# ─────────────────────────────────────────────
# GEOMETRY HELPERS (same as OBJ generator)
# ─────────────────────────────────────────────

def make_sphere(cx, cy, cz, rx, ry, rz, n_lat=14, n_lon=20,
                lat_min=0.0, lat_max=math.pi,
                lon_min=0.0, lon_max=2*math.pi):
    verts, norms, faces = [], [], []
    for i in range(n_lat + 1):
        lat = lat_min + (lat_max - lat_min) * i / n_lat
        for j in range(n_lon + 1):
            lon = lon_min + (lon_max - lon_min) * j / n_lon
            nx = math.sin(lat) * math.sin(lon)
            ny = math.cos(lat)
            nz = math.sin(lat) * math.cos(lon)
            verts.append((cx + rx * nx, cy + ry * ny, cz + rz * nz))
            length = math.sqrt(nx*nx + ny*ny + nz*nz) or 1
            norms.append((nx/length, ny/length, nz/length))
    cols = n_lon + 1
    for i in range(n_lat):
        for j in range(n_lon):
            v0 = i * cols + j
            v1 = v0 + 1
            v2 = (i + 1) * cols + j
            v3 = v2 + 1
            faces.append((v0, v2, v1))
            faces.append((v1, v2, v3))
    return verts, norms, faces


def make_cylinder(cx, y_bot, cz, r_bot, r_top, height, n_seg=20, n_rows=1,
                  lon_min=0.0, lon_max=2*math.pi, cap_top=False, cap_bot=False):
    verts, norms, faces = [], [], []
    for i in range(n_rows + 1):
        t = i / n_rows
        y = y_bot + height * t
        r = r_bot + (r_top - r_bot) * t
        for j in range(n_seg + 1):
            angle = lon_min + (lon_max - lon_min) * j / n_seg
            nx = math.sin(angle)
            nz = math.cos(angle)
            slope = (r_bot - r_top) / (height if height > 0 else 1)
            ny_n = slope
            length = math.sqrt(nx*nx + ny_n*ny_n + nz*nz) or 1
            verts.append((cx + r * nx, y, cz + r * nz))
            norms.append((nx/length, ny_n/length, nz/length))
    cols = n_seg + 1
    for i in range(n_rows):
        for j in range(n_seg):
            v0 = i * cols + j; v1 = v0 + 1
            v2 = (i + 1) * cols + j; v3 = v2 + 1
            faces.append((v0, v2, v1))
            faces.append((v1, v2, v3))
    if cap_bot:
        ci = len(verts)
        verts.append((cx, y_bot, cz)); norms.append((0, -1, 0))
        for j in range(n_seg):
            a1 = lon_min + (lon_max - lon_min) * j / n_seg
            a2 = lon_min + (lon_max - lon_min) * (j+1) / n_seg
            v1i = len(verts)
            verts.append((cx + r_bot*math.sin(a1), y_bot, cz + r_bot*math.cos(a1)))
            norms.append((0, -1, 0))
            v2i = len(verts)
            verts.append((cx + r_bot*math.sin(a2), y_bot, cz + r_bot*math.cos(a2)))
            norms.append((0, -1, 0))
            faces.append((ci, v2i, v1i))
    if cap_top:
        y_t = y_bot + height
        ci = len(verts)
        verts.append((cx, y_t, cz)); norms.append((0, 1, 0))
        for j in range(n_seg):
            a1 = lon_min + (lon_max - lon_min) * j / n_seg
            a2 = lon_min + (lon_max - lon_min) * (j+1) / n_seg
            v1i = len(verts)
            verts.append((cx + r_top*math.sin(a1), y_t, cz + r_top*math.cos(a1)))
            norms.append((0, 1, 0))
            v2i = len(verts)
            verts.append((cx + r_top*math.sin(a2), y_t, cz + r_top*math.cos(a2)))
            norms.append((0, 1, 0))
            faces.append((ci, v1i, v2i))
    return verts, norms, faces


def make_rounded_box(cx, cy, cz, sx, sy, sz, n_seg=8):
    return make_sphere(cx, cy, cz, sx, sy, sz, n_lat=10, n_lon=16)


def make_limb(x1, y1, z1, x2, y2, z2, r1, r2, n_seg=12, n_rows=6):
    dx, dy, dz = x2-x1, y2-y1, z2-z1
    length = math.sqrt(dx*dx + dy*dy + dz*dz)
    if length < 1e-6: return [], [], []
    up = (dx/length, dy/length, dz/length)
    ref = (0, 1, 0) if abs(up[1]) < 0.9 else (1, 0, 0)
    right = (up[1]*ref[2]-up[2]*ref[1], up[2]*ref[0]-up[0]*ref[2], up[0]*ref[1]-up[1]*ref[0])
    rl = math.sqrt(sum(c**2 for c in right)) or 1
    right = tuple(c/rl for c in right)
    fwd = (up[1]*right[2]-up[2]*right[1], up[2]*right[0]-up[0]*right[2], up[0]*right[1]-up[1]*right[0])

    verts, norms, faces = [], [], []
    for i in range(n_rows + 1):
        t = i / n_rows
        px, py, pz = x1+dx*t, y1+dy*t, z1+dz*t
        r = r1 + (r2-r1)*t
        for j in range(n_seg + 1):
            angle = 2*math.pi*j/n_seg
            ca, sa = math.cos(angle), math.sin(angle)
            nx = right[0]*ca + fwd[0]*sa
            ny = right[1]*ca + fwd[1]*sa
            nz = right[2]*ca + fwd[2]*sa
            verts.append((px+r*nx, py+r*ny, pz+r*nz))
            norms.append((nx, ny, nz))
    cols = n_seg + 1
    for i in range(n_rows):
        for j in range(n_seg):
            v0 = i*cols+j; v1 = v0+1; v2 = (i+1)*cols+j; v3 = v2+1
            faces.append((v0, v2, v1)); faces.append((v1, v2, v3))
    # Bottom cap
    ci = len(verts)
    verts.append((x1, y1, z1)); norms.append((-up[0], -up[1], -up[2]))
    for j in range(n_seg):
        faces.append((ci, j, (j+1) % n_seg))
    # Top cap
    ci = len(verts)
    verts.append((x2, y2, z2)); norms.append(up)
    top_start = n_rows * cols
    for j in range(n_seg):
        faces.append((ci, top_start + (j+1)%n_seg, top_start + j))
    return verts, norms, faces


# ─────────────────────────────────────────────
# BODY REGION BUILDER
# ─────────────────────────────────────────────

def build_body_regions():
    regions = {}
    head_cx, head_cy, head_cz = 0, 1.62, 0
    head_r = 0.105

    # Scalp
    regions['region_scalp'] = make_sphere(
        head_cx, head_cy, head_cz, head_r, head_r*1.05, head_r,
        n_lat=12, n_lon=20, lat_min=0, lat_max=math.pi*0.45)

    # Face
    regions['region_face'] = make_sphere(
        head_cx, head_cy-0.01, head_cz+0.005,
        head_r*0.95, head_r*0.7, head_r*0.9,
        n_lat=10, n_lon=16,
        lat_min=math.pi*0.3, lat_max=math.pi*0.85,
        lon_min=-math.pi*0.55, lon_max=math.pi*0.55)

    # Ears
    ear_v, ear_n, ear_f = [], [], []
    for side in [-1, 1]:
        ev, en, ef = make_sphere(
            head_cx + side*0.115, head_cy+0.01, head_cz-0.01,
            0.025, 0.04, 0.015, n_lat=8, n_lon=10)
        off = len(ear_v)
        ear_v.extend(ev); ear_n.extend(en)
        ear_f.extend([(f[0]+off, f[1]+off, f[2]+off) for f in ef])
    regions['region_ear'] = (ear_v, ear_n, ear_f)

    # Neck
    regions['region_neck'] = make_cylinder(
        0, 1.44, 0, 0.055, 0.06, 0.08, n_seg=16, n_rows=3, cap_top=True, cap_bot=True)

    # Chest
    regions['region_chest'] = make_cylinder(
        0, 1.18, 0, 0.17, 0.20, 0.26, n_seg=20, n_rows=5,
        lon_min=-math.pi*0.48, lon_max=math.pi*0.48, cap_top=True, cap_bot=True)

    # Abdomen
    regions['region_abdomen'] = make_cylinder(
        0, 0.94, 0, 0.155, 0.17, 0.24, n_seg=20, n_rows=4,
        lon_min=-math.pi*0.48, lon_max=math.pi*0.48, cap_top=False, cap_bot=True)

    # Back
    regions['region_back'] = make_cylinder(
        0, 0.92, 0, 0.16, 0.19, 0.52, n_seg=20, n_rows=6,
        lon_min=math.pi*0.48, lon_max=math.pi*1.52, cap_top=True, cap_bot=True)

    # Arms (upper extremity)
    arm_v, arm_n, arm_f = [], [], []
    for side in [-1, 1]:
        for (x1,y1,z1,x2,y2,z2,r1,r2) in [
            (side*0.24,1.38,0, side*0.32,1.12,-0.01, 0.048,0.040),
            (side*0.32,1.12,-0.01, side*0.38,0.88,0.01, 0.038,0.030),
        ]:
            sv, sn, sf = make_limb(x1,y1,z1,x2,y2,z2,r1,r2, n_seg=12, n_rows=6)
            off = len(arm_v)
            arm_v.extend(sv); arm_n.extend(sn)
            arm_f.extend([(f[0]+off, f[1]+off, f[2]+off) for f in sf])
    regions['region_upper_extremity'] = (arm_v, arm_n, arm_f)

    # Hands
    hand_v, hand_n, hand_f = [], [], []
    for side in [-1, 1]:
        hv, hn, hf = make_rounded_box(side*0.40, 0.82, 0.01, 0.030, 0.050, 0.018, n_seg=8)
        off = len(hand_v)
        hand_v.extend(hv); hand_n.extend(hn)
        hand_f.extend([(f[0]+off, f[1]+off, f[2]+off) for f in hf])
    regions['region_hand'] = (hand_v, hand_n, hand_f)

    # Legs (lower extremity)
    leg_v, leg_n, leg_f = [], [], []
    for side in [-1, 1]:
        for (x1,y1,z1,x2,y2,z2,r1,r2,ns,nr) in [
            (side*0.10,0.92,0, side*0.10,0.50,0, 0.070,0.050, 14,7),
            (side*0.10,0.50,0, side*0.10,0.08,0.02, 0.045,0.035, 12,6),
        ]:
            sv, sn, sf = make_limb(x1,y1,z1,x2,y2,z2,r1,r2, n_seg=ns, n_rows=nr)
            off = len(leg_v)
            leg_v.extend(sv); leg_n.extend(sn)
            leg_f.extend([(f[0]+off, f[1]+off, f[2]+off) for f in sf])
    regions['region_lower_extremity'] = (leg_v, leg_n, leg_f)

    # Feet
    foot_v, foot_n, foot_f = [], [], []
    for side in [-1, 1]:
        fv, fn, ff = make_sphere(side*0.10, 0.04, 0.04, 0.04, 0.035, 0.07, n_lat=8, n_lon=12)
        off = len(foot_v)
        foot_v.extend(fv); foot_n.extend(fn)
        foot_f.extend([(f[0]+off, f[1]+off, f[2]+off) for f in ff])
    regions['region_foot'] = (foot_v, foot_n, foot_f)

    # Genital
    regions['region_genital'] = make_sphere(
        0, 0.92, 0.10, 0.05, 0.035, 0.03, n_lat=8, n_lon=10,
        lat_min=math.pi*0.2, lat_max=math.pi*0.8,
        lon_min=-math.pi*0.4, lon_max=math.pi*0.4)

    return regions


# ─────────────────────────────────────────────
# COLLADA (.dae) WRITER
# ─────────────────────────────────────────────

COLLADA_NS = "http://www.collada.org/2005/11/COLLADASchema"

def floats_to_str(floats):
    """Flatten list of tuples to space-separated string."""
    parts = []
    for item in floats:
        if isinstance(item, (tuple, list)):
            parts.extend(f"{v:.6f}" for v in item)
        else:
            parts.append(f"{item:.6f}")
    return " ".join(parts)

def ints_to_str(indices):
    """Flatten triangle index tuples for COLLADA <p> element."""
    parts = []
    for tri in indices:
        for idx in tri:
            # position index and normal index are the same (shared)
            parts.append(str(idx))
            parts.append(str(idx))
    return " ".join(parts)


def write_dae(regions, filepath):
    """Write a complete COLLADA .dae file with per-region nodes."""

    # Default body color: neutral medical gray
    body_color = (0.78, 0.78, 0.80, 1.0)
    ambient_color = (0.15, 0.15, 0.16, 1.0)
    specular_color = (0.20, 0.20, 0.20, 1.0)

    region_names = [
        'region_scalp', 'region_face', 'region_ear', 'region_neck',
        'region_chest', 'region_abdomen', 'region_back',
        'region_upper_extremity', 'region_lower_extremity',
        'region_hand', 'region_foot', 'region_genital'
    ]

    lines = []
    lines.append('<?xml version="1.0" encoding="utf-8"?>')
    lines.append('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">')

    # ── Asset ──
    lines.append('  <asset>')
    lines.append('    <contributor>')
    lines.append('      <author>DermaFusion Generator</author>')
    lines.append('      <authoring_tool>DermaFusion Body Model Generator</authoring_tool>')
    lines.append('    </contributor>')
    lines.append('    <created>2026-02-14T00:00:00</created>')
    lines.append('    <modified>2026-02-14T00:00:00</modified>')
    lines.append('    <unit name="meter" meter="1"/>')
    lines.append('    <up_axis>Y_UP</up_axis>')
    lines.append('  </asset>')

    # ── Effects ──
    lines.append('  <library_effects>')
    for name in region_names:
        eid = f"effect_{name}"
        lines.append(f'    <effect id="{eid}">')
        lines.append(f'      <profile_COMMON>')
        lines.append(f'        <technique sid="common">')
        lines.append(f'          <phong>')
        lines.append(f'            <ambient>')
        lines.append(f'              <color>{ambient_color[0]} {ambient_color[1]} {ambient_color[2]} {ambient_color[3]}</color>')
        lines.append(f'            </ambient>')
        lines.append(f'            <diffuse>')
        lines.append(f'              <color>{body_color[0]} {body_color[1]} {body_color[2]} {body_color[3]}</color>')
        lines.append(f'            </diffuse>')
        lines.append(f'            <specular>')
        lines.append(f'              <color>{specular_color[0]} {specular_color[1]} {specular_color[2]} {specular_color[3]}</color>')
        lines.append(f'            </specular>')
        lines.append(f'            <shininess><float>30.0</float></shininess>')
        lines.append(f'            <transparency><float>1.0</float></transparency>')
        lines.append(f'          </phong>')
        lines.append(f'        </technique>')
        lines.append(f'      </profile_COMMON>')
        lines.append(f'    </effect>')
    lines.append('  </library_effects>')

    # ── Materials ──
    lines.append('  <library_materials>')
    for name in region_names:
        mid = f"material_{name}"
        eid = f"effect_{name}"
        lines.append(f'    <material id="{mid}" name="{mid}">')
        lines.append(f'      <instance_effect url="#{eid}"/>')
        lines.append(f'    </material>')
    lines.append('  </library_materials>')

    # ── Geometries ──
    lines.append('  <library_geometries>')
    for name in region_names:
        verts, norms, faces = regions[name]
        gid = f"geometry_{name}"
        n_verts = len(verts)
        n_faces = len(faces)

        pos_str = floats_to_str(verts)
        norm_str = floats_to_str(norms)

        # Build <p> element: for each triangle, interleave pos_idx and norm_idx
        p_parts = []
        for tri in faces:
            for idx in tri:
                p_parts.append(str(idx))  # VERTEX (position) index
                p_parts.append(str(idx))  # NORMAL index (same)
        p_str = " ".join(p_parts)

        lines.append(f'    <geometry id="{gid}" name="{name}">')
        lines.append(f'      <mesh>')

        # Positions source
        lines.append(f'        <source id="{gid}-positions">')
        lines.append(f'          <float_array id="{gid}-positions-array" count="{n_verts * 3}">{pos_str}</float_array>')
        lines.append(f'          <technique_common>')
        lines.append(f'            <accessor source="#{gid}-positions-array" count="{n_verts}" stride="3">')
        lines.append(f'              <param name="X" type="float"/>')
        lines.append(f'              <param name="Y" type="float"/>')
        lines.append(f'              <param name="Z" type="float"/>')
        lines.append(f'            </accessor>')
        lines.append(f'          </technique_common>')
        lines.append(f'        </source>')

        # Normals source
        lines.append(f'        <source id="{gid}-normals">')
        lines.append(f'          <float_array id="{gid}-normals-array" count="{n_verts * 3}">{norm_str}</float_array>')
        lines.append(f'          <technique_common>')
        lines.append(f'            <accessor source="#{gid}-normals-array" count="{n_verts}" stride="3">')
        lines.append(f'              <param name="X" type="float"/>')
        lines.append(f'              <param name="Y" type="float"/>')
        lines.append(f'              <param name="Z" type="float"/>')
        lines.append(f'            </accessor>')
        lines.append(f'          </technique_common>')
        lines.append(f'        </source>')

        # Vertices
        lines.append(f'        <vertices id="{gid}-vertices">')
        lines.append(f'          <input semantic="POSITION" source="#{gid}-positions"/>')
        lines.append(f'        </vertices>')

        # Triangles
        lines.append(f'        <triangles material="material_{name}" count="{n_faces}">')
        lines.append(f'          <input semantic="VERTEX" source="#{gid}-vertices" offset="0"/>')
        lines.append(f'          <input semantic="NORMAL" source="#{gid}-normals" offset="1"/>')
        lines.append(f'          <p>{p_str}</p>')
        lines.append(f'        </triangles>')

        lines.append(f'      </mesh>')
        lines.append(f'    </geometry>')

    lines.append('  </library_geometries>')

    # ── Cameras ──
    lines.append('  <library_cameras>')
    lines.append('    <camera id="Camera" name="Camera">')
    lines.append('      <optics>')
    lines.append('        <technique_common>')
    lines.append('          <perspective>')
    lines.append('            <yfov>45</yfov>')
    lines.append('            <znear>0.01</znear>')
    lines.append('            <zfar>100</zfar>')
    lines.append('          </perspective>')
    lines.append('        </technique_common>')
    lines.append('      </optics>')
    lines.append('    </camera>')
    lines.append('  </library_cameras>')

    # ── Lights ──
    lines.append('  <library_lights>')
    # Ambient light
    lines.append('    <light id="AmbientLight" name="AmbientLight">')
    lines.append('      <technique_common>')
    lines.append('        <ambient>')
    lines.append('          <color>0.4 0.4 0.42 1</color>')
    lines.append('        </ambient>')
    lines.append('      </technique_common>')
    lines.append('    </light>')
    # Directional light (key light from upper-front-right)
    lines.append('    <light id="DirectionalLight" name="DirectionalLight">')
    lines.append('      <technique_common>')
    lines.append('        <directional>')
    lines.append('          <color>0.85 0.85 0.87 1</color>')
    lines.append('        </directional>')
    lines.append('      </technique_common>')
    lines.append('    </light>')
    # Fill light from left
    lines.append('    <light id="FillLight" name="FillLight">')
    lines.append('      <technique_common>')
    lines.append('        <directional>')
    lines.append('          <color>0.3 0.3 0.32 1</color>')
    lines.append('        </directional>')
    lines.append('      </technique_common>')
    lines.append('    </light>')
    lines.append('  </library_lights>')

    # ── Visual Scene ──
    lines.append('  <library_visual_scenes>')
    lines.append('    <visual_scene id="Scene" name="DermaFusion_BodyModel">')

    # Camera node - positioned to frame the full body
    lines.append('      <node id="CameraNode" name="CameraNode" type="NODE">')
    lines.append('        <translate>0 0.85 2.8</translate>')
    lines.append('        <rotate sid="rotateX">1 0 0 -5</rotate>')
    lines.append('        <instance_camera url="#Camera"/>')
    lines.append('      </node>')

    # Light nodes
    lines.append('      <node id="AmbientLightNode" name="AmbientLightNode" type="NODE">')
    lines.append('        <instance_light url="#AmbientLight"/>')
    lines.append('      </node>')

    lines.append('      <node id="DirectionalLightNode" name="DirectionalLightNode" type="NODE">')
    lines.append('        <translate>2 3 2</translate>')
    lines.append('        <rotate>1 0 0 -35</rotate>')
    lines.append('        <rotate>0 1 0 30</rotate>')
    lines.append('        <instance_light url="#DirectionalLight"/>')
    lines.append('      </node>')

    lines.append('      <node id="FillLightNode" name="FillLightNode" type="NODE">')
    lines.append('        <translate>-1.5 2 1</translate>')
    lines.append('        <rotate>1 0 0 -25</rotate>')
    lines.append('        <rotate>0 1 0 -40</rotate>')
    lines.append('        <instance_light url="#FillLight"/>')
    lines.append('      </node>')

    # Body region nodes — these are the named nodes SceneKit finds via childNode(withName:)
    for name in region_names:
        gid = f"geometry_{name}"
        mid = f"material_{name}"
        lines.append(f'      <node id="{name}" name="{name}" type="NODE">')
        lines.append(f'        <instance_geometry url="#{gid}">')
        lines.append(f'          <bind_material>')
        lines.append(f'            <technique_common>')
        lines.append(f'              <instance_material symbol="{mid}" target="#{mid}"/>')
        lines.append(f'            </technique_common>')
        lines.append(f'          </bind_material>')
        lines.append(f'        </instance_geometry>')
        lines.append(f'      </node>')

    lines.append('    </visual_scene>')
    lines.append('  </library_visual_scenes>')

    # ── Scene reference ──
    lines.append('  <scene>')
    lines.append('    <instance_visual_scene url="#Scene"/>')
    lines.append('  </scene>')

    lines.append('</COLLADA>')

    # Write
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))

    # Stats
    total_v = sum(len(regions[n][0]) for n in region_names)
    total_f = sum(len(regions[n][2]) for n in region_names)
    fsize = os.path.getsize(filepath)
    print(f"Wrote {filepath}")
    print(f"  {total_v} vertices, {total_f} triangles, 12 regions")
    print(f"  File size: {fsize / 1024:.1f} KB")
    print(f"  Camera: positioned at (0, 0.85, 2.8) to frame full body")
    print(f"  Lights: ambient + directional key + fill (3-point)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    output_dir = '/home/claude/output'
    os.makedirs(output_dir, exist_ok=True)

    print("Generating DermaFusion COLLADA body model...")
    regions = build_body_regions()

    dae_path = os.path.join(output_dir, 'BodyModel.dae')
    write_dae(regions, dae_path)

    print("\n── Node Names (for SceneKit hit testing) ──")
    for name in [
        'region_scalp', 'region_face', 'region_ear', 'region_neck',
        'region_chest', 'region_abdomen', 'region_back',
        'region_upper_extremity', 'region_lower_extremity',
        'region_hand', 'region_foot', 'region_genital'
    ]:
        v, n, f = regions[name]
        print(f"  {name:35s}  {len(v):5d} verts  {len(f):5d} tris")

    print("\n── Xcode Integration ──")
    print("1. Drag BodyModel.dae into Xcode project → Resources/")
    print("2. Select the file in Xcode → Editor > Convert to SceneKit Scene File Format (.scn)")
    print("3. Rename to BodyModel.scn (or keep .dae — SCNScene loads both)")
    print("4. Delete the original .dae after conversion")
    print("")
    print("In Swift:")
    print('  let scene = SCNScene(named: "BodyModel.scn")  // or "BodyModel.dae"')
    print('  let scalp = scene?.rootNode.childNode(withName: "region_scalp", recursively: true)')
