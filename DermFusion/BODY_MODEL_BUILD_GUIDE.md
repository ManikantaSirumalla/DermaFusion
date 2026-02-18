# DermaFusion 3D Body Model — Complete Build Guide

## What You Need to Produce

A SceneKit-compatible 3D file (`BodyModel.scn`) containing a realistic human body mesh split into **12 separately named nodes**. Each node must have skin-toned materials with proper UVs. The iOS app detects taps via `hitTest`, identifies the node name, and highlights the selected region.

**Required node names (exact):**
```
region_scalp              region_neck
region_face               region_chest
region_ear                region_abdomen
region_neck               region_back
region_upper_extremity    region_hand
region_lower_extremity    region_foot
region_genital
```

---

## Prerequisites

**Install these before starting (all free):**

| Tool | Purpose | Download |
|------|---------|----------|
| **Blender 4.0+** | 3D modeling, mesh segmentation, export | https://www.blender.org/download/ |
| **MakeHuman** or **MPFB2** | Generate realistic human mesh | See Step 1 below |
| **Apple Reality Converter** | Convert .glb → .usdz (Mac only) | https://developer.apple.com/augmented-reality/tools/ |
| **Xcode 15+** | Convert .dae/.usdz → .scn | Already installed |

**Time estimate:** 2–3 hours for first run (including installs). Under 30 minutes once familiar.

---

## Step 1: Generate a Realistic Human Mesh

You have two options. **Option A is recommended** (faster, stays inside Blender).

### Option A: MPFB2 Add-on (Inside Blender)

MPFB2 is MakeHuman rebuilt as a Blender add-on. One tool, no separate app.

1. Open Blender → Edit → Preferences → Add-ons → Get Extensions
2. Search **"MPFB"** → Install the MakeHuman Community add-on
3. In the 3D viewport sidebar (press N), find the **MPFB** tab
4. Click **"Create Human"** under the New Human panel
5. Configure the character:
   - Set **Gender** slider to ~0.5 (androgynous medical mannequin)
   - Set **Age** to ~0.5 (adult)
   - Set **Muscle** to ~0.3 (light definition, not bodybuilder)
   - Set **Weight** to ~0.3 (average build)
   - Leave skin/eyes as defaults
6. Click **"Create"** — a full human mesh appears with skin texture and UVs

**Result:** A ~13,000-vertex human mesh with proper topology, skin material, facial features, UV mapping, and textures. All under **CC0 license** — free to use in your app without attribution.

### Option B: MakeHuman Standalone App

1. Download MakeHuman from https://static.makehumancommunity.org/makehuman.html
2. Launch, adjust sliders (gender, age, weight, muscle, ethnicity)
3. Go to **Files → Export** tab
4. Select format: **Collada (DAE)** with "Feet on Ground" checked
5. Export to a known folder
6. In Blender: **File → Import → Collada (.dae)** → select the exported file

**Result:** Same quality mesh, but requires a separate app install.

### What You Should See

After either option, you should have in Blender:
- A realistic human mesh (single object or small group of objects)
- Skin-toned material with texture
- Proper UV unwrapping
- T-pose or A-pose (arms out from body)
- Facing forward (typically -Y in Blender's coordinate system)

**Important:** If the mesh came as multiple objects (body, eyes, eyebrows, etc.), select all mesh parts and press **Ctrl+J** to join into a single object. Delete any non-body parts (hair, teeth, tongue, eyeballs) — they'll interfere with region segmentation. Keep only the body/skin mesh.

---

## Step 2: Prepare the Mesh

Before running the segmentation script:

### 2a. Clean Up

1. Select the body mesh
2. **Tab** into Edit Mode
3. Select all (A), then **Mesh → Clean Up → Delete Loose** (removes stray vertices)
4. **Mesh → Normals → Recalculate Outside** (fixes inverted faces)
5. **Tab** back to Object Mode

### 2b. Apply Transforms

1. Select the body mesh
2. **Ctrl+A → All Transforms** (Apply location, rotation, scale)
3. This ensures the script reads correct world-space coordinates

### 2c. Verify Orientation

The script auto-detects the up axis, but verify:
- The model should be **standing upright** (tallest dimension = height)
- Arms should be **out to the sides** (T-pose or A-pose)
- Facing the camera when you press Numpad 1 (front view)

---

## Step 3: Run the Segmentation Script

This is the automated step. The provided `dermafusion_segment_body.py` script:

1. Analyzes the mesh bounding box to determine body proportions
2. Classifies every face into one of 12 body regions by vertex position
3. Separates the mesh into 12 named objects
4. Applies a skin material
5. Adds camera and 3-point lighting
6. Exports as .dae, .obj, and .glb

### How to Run

1. Select the body mesh in the viewport (click on it — orange outline)
2. Open the **Scripting** workspace (top menu bar tabs)
3. Click **New** in the text editor
4. Paste the entire contents of `dermafusion_segment_body.py`
5. **Adjust `EXPORT_PATH`** at the top of the script to your preferred location:
   ```python
   EXPORT_PATH = os.path.expanduser("~/Desktop/BodyModel.dae")
   ```
6. Click **Run Script** (or press Alt+P)

### What Happens

The console (Window → Toggle System Console on Windows, or terminal on Mac) shows:

```
============================================================
DermaFusion Body Segmenter
============================================================
Input mesh: 'Human' (25834 faces, 13377 verts)
  Bounding box: 1.628 x 1.733 x 0.292
  Up axis: Y, Right axis: X, Forward axis: Z
  Height: 1.733, Width: 1.628, Depth: 0.292

Face classification:
  region_abdomen                        2847 faces
  region_back                           4521 faces
  region_chest                          3012 faces
  region_ear                             342 faces
  ...

  Created: region_scalp (1245 verts, 2103 faces)
  Created: region_face (876 verts, 1534 faces)
  ...

VERIFICATION
============================================================
  ✓ region_scalp                         1245 verts   2103 faces  mat:✓  uv:✓
  ✓ region_face                           876 verts   1534 faces  mat:✓  uv:✓
  ...
  ✓ ALL 12 REGIONS PRESENT — Ready for SceneKit
```

### If Regions Are Missing or Wrong

The script classifies faces by position relative to the bounding box. If your mesh has unusual proportions (very muscular arms, unusual pose), some thresholds may need adjustment.

Open the script and find `classify_vertex_region()`. The key thresholds:

```python
# Height thresholds (0 = feet, 1 = top of head)
h > 0.88    # Head region starts here
h > 0.82    # Neck starts here
h > 0.50    # Torso/leg boundary
h < 0.05    # Feet

# Left-right thresholds (distance from center, 0 = center, 0.5 = edge)
lr_dist > 0.18    # Arms start here (adjusts for T-pose vs A-pose)
lr_dist > 0.42    # Hands start here

# Front-back threshold
fb < 0.40    # Back (behind this line)
fb > 0.45    # Front (ahead of this line)
```

Adjust these if a region is capturing too many or too few faces. After editing, re-run the script (it cleans up previous runs automatically).

---

## Step 4: Visual Inspection

Before exporting, visually verify the segmentation:

1. In the 3D viewport, select each `region_*` object one at a time
2. Press **H** to hide it, verifying the correct body area disappears
3. Press **Alt+H** to unhide all
4. Check that:
   - Scalp covers the top of the head
   - Face covers the front face area
   - Ears are isolated on both sides
   - Chest/abdomen split is at roughly the navel line
   - Back covers the entire posterior
   - Arms run from shoulder to wrist
   - Hands are isolated
   - Legs run from hip to ankle
   - Feet are isolated
   - Genital region is a small discrete area

### Quick Fix: Reassign Faces Manually

If a few faces are in the wrong region:

1. Select the source object (where faces are currently)
2. Tab into Edit Mode
3. Select the misassigned faces (use face select mode: press 3)
4. **P → Selection** (Separate selected faces)
5. Tab back to Object Mode
6. Select the new separated chunk
7. Then Shift-select the target region object
8. **Ctrl+J** (Join into the target)

---

## Step 5: Export for Xcode

The script exports automatically, but if you need to re-export manually:

### Option A: COLLADA (.dae) — Direct to SceneKit

1. Select all region objects + camera + lights (A to select all, or box select)
2. **File → Export → Collada (.dae)**
3. In the export settings:
   - Check **"Selection Only"**
   - Check **"Triangulate"**
   - Check **"Include Material Textures"**
4. Save as `BodyModel.dae`

### Option B: glTF → USDZ → .scn — Best Quality

This pipeline preserves materials and textures most faithfully:

1. Select all objects → **File → Export → glTF 2.0 (.glb)**
   - Format: **glTF Binary (.glb)**
   - Check **"Selected Objects"**
   - Check **"Apply Modifiers"**
   - Materials: **Export**
2. Open **Apple Reality Converter** (Mac)
3. **File → Import** → select the `.glb` file
4. Verify the model looks correct (rotate, check materials)
5. **File → Export** → save as `BodyModel.usdz`
6. Drag `BodyModel.usdz` into Xcode
7. Select it → **Editor → Convert to SceneKit Scene File Format (.scn)**
8. Rename to `BodyModel.scn`

### Option C: OBJ — Simple Fallback

The `.obj` + `.mtl` pair also works. Drag both into Xcode's Resources. SceneKit loads OBJ directly:

```swift
let scene = SCNScene(named: "BodyModel.obj")
```

---

## Step 6: Xcode Integration

1. Drag your exported file into the Xcode project under `Resources/`
2. If `.dae` or `.usdz`: select it → **Editor → Convert to SceneKit Scene File Format (.scn)**
3. Open the Scene Editor and verify the scene graph shows all 12 `region_*` nodes
4. Add `BodyModelVerificationTests.swift` to your test target
5. Run tests to confirm all nodes load correctly:

```
Test Suite 'BodyModelVerificationTests' passed
  testBodyModelSceneLoads - passed
  testAllTwelveRegionNodesExist - passed (12/12 found)
  testRegionNodesHaveReasonableGeometry - passed
  testSceneHasCameraAndLights - passed
  testHitTestReturnsRegionNames - passed
```

---

## Step 7: Verify on Device

The true test is running on a real device (or Simulator):

1. Build and run the app
2. Navigate to the body map screen
3. Verify:
   - Model renders with skin material (not wireframe, not black)
   - Orbital rotation works (drag to spin)
   - Tap on each region → correct region highlights
   - Region name label updates correctly
   - "Confirm" button activates after selection
4. Test edge cases:
   - Tap between regions (should select the one hit first)
   - Rapid taps (should not crash or double-select)
   - Very fast rotation + tap (hit test should still work)

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Model is black/untextured | Materials didn't export | Re-export with "Include Material Textures" checked |
| Model is invisible | Wrong coordinate system | Check Y-up vs Z-up in export settings |
| Nodes missing after .scn conversion | Name truncation | Check node names in Scene Editor — Xcode sometimes appends suffixes |
| Hit test returns nil | Tap on gap between regions | Make regions overlap slightly, or increase hit test tolerance |
| Texture is stretched | UV mapping lost in export | Use glTF pipeline (Option B) — preserves UVs best |
| Model too large/small | Scale mismatch | In Blender, apply scale (Ctrl+A) before export; or adjust SCNNode scale in code |
| Ears not detected | Too small | Lower the `lr_dist > 0.08` threshold in classify_vertex_region() |
| MakeHuman not generating via MPFB | Add-on needs content packs | Download base assets from MPFB documentation |

---

## License Summary

| Component | License | Attribution Required? |
|-----------|---------|----------------------|
| MakeHuman exported mesh | CC0 | No |
| MPFB2-generated mesh | CC0 | No |
| MakeHuman skin textures | CC0 | No |
| Blender segmentation script | CC0 | No |
| Your modifications | Yours | N/A |

You can ship the model in a commercial app with zero attribution requirements.

---

## File Checklist

After completing all steps, you should have:

```
DermaFusion-iOS/
├── Resources/
│   ├── BodyModel.scn          ← Final SceneKit scene (from Step 6)
│   └── (BodyModel.usdz)       ← Optional: keep as reference
├── Views/Tab1_Scan/
│   ├── BodyMapView.swift       ← SwiftUI wrapper
│   └── BodyMap3DView.swift     ← UIViewRepresentable (SCNView + hitTest)
├── ViewModels/
│   └── BodyMapViewModel.swift  ← Selection state, region name display
└── Tests/
    └── BodyModelVerificationTests.swift  ← Automated node checks
```

The Blender project file and scripts stay in your development tools — they don't ship with the app.
