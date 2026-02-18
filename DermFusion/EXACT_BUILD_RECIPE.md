# DermaFusion Body Model — Exact Build Recipe

## Reference Target

Realistic male anatomical mesh with:
- Visible clean mesh topology (edge loops)
- Matte skin material (warm flesh tone)
- Facial features (eyes, nose, mouth, brow ridge)
- Athletic/moderate muscular definition
- A-pose (arms ~30° from body, palms forward)
- Bald, no clothing, no accessories
- Clinical/medical aesthetic

---

## Step 1: Install MPFB2 + Asset Packs

### 1a. Install the Add-on

1. Open **Blender 4.2+**
2. Edit → Preferences → **Get Extensions**
3. Search: **"MPFB"**
4. Click **Install** on "MPFB" by MakeHuman Community
5. Close Preferences

### 1b. Install Asset Packs (Required for Skin)

Without asset packs you get an untextured gray mesh. You need at minimum
the **"MakeHuman System Assets"** pack for skin materials.

1. Go to: https://static.makehumancommunity.org/mpfb/downloads/asset_packs.html
2. Download **"makehuman_system_assets"** (contains skins, eyes, teeth)
3. Optionally download **"skins"** pack for more skin options
4. In Blender: Edit → Preferences → Add-ons → find MPFB
5. Expand MPFB preferences → click **"Install Asset Pack"** 
6. Select the downloaded .zip file(s)
7. **Restart Blender** after installing packs

---

## Step 2: Create the Human Mesh

### 2a. Open MPFB Panel

1. In the 3D viewport, press **N** to open the sidebar
2. Click the **MPFB** tab (if not visible, the add-on isn't enabled)

### 2b. Configure "From Scratch" Settings

Go to **New Human → From Scratch** panel. Set these values to match
the reference image:

```
Phenotype Sliders:
  Gender:     1.0    (fully male)
  Age:        0.5    (young adult, ~30 years)
  Muscle:     0.6    (athletic definition — NOT bodybuilder)
  Weight:     0.3    (lean build, abs visible)
  Height:     0.5    (average, ~175cm)
  
  Proportions:  0.5  (default/ideal)
  
  African:    0.0
  Asian:      0.0
  Caucasian:  1.0    (matches reference image skin tone)
```

> **Why these values:** Gender 1.0 gives male bone structure and musculature.
> Muscle 0.6 gives the visible-but-not-extreme definition shown in the reference.
> Weight 0.3 keeps the mesh lean so ab definition and ribcage are visible.
> The reference image shows a Caucasian body type.

### 2c. Create the Mesh

1. Click **"Create Human"**
2. A mesh appears in the viewport — it will be untextured (gray) at first
3. This is the MakeHuman base mesh: ~13,000 vertices with proper
   edge loop topology, UV unwrapping, and facial feature geometry

### 2d. Fine-Tune (Optional)

With the human selected, go to the **Model** panel in the MPFB sidebar.
The sub-panels let you adjust hundreds of parameters:

```
Recommended tweaks:
  Torso → Torso Muscles:  0.6  (enhances pectoral/ab definition)
  Arms  → Upper Arm Size: 0.5  (slightly bulkier upper arms)  
  Face  → Brow Ridge:     0.5  (visible brow like reference)
  Face  → Jaw Width:      0.4  (defined jawline)
  Face  → Chin Size:      0.5  (neutral)
```

The defaults are already close to the reference — don't over-tweak.

---

## Step 3: Apply Skin Material

### 3a. Load a Skin from Library

1. With the human mesh selected
2. In MPFB sidebar → **Apply Assets → Skin Library**
3. Browse available skins — look for one labeled similar to:
   - "young_caucasian_male" 
   - "middleage_caucasian_male"
   - Or any skin with a warm flesh tone
4. Click **"Load"** on your chosen skin

The mesh now has a procedural skin material with:
- Diffuse skin color map
- Subsurface scattering (realistic skin translucency)
- Normal map (pore-level detail)
- Specular/roughness maps

### 3b. Adjust Skin for Matte Medical Look

The reference image has a **matte** skin — less shiny than a typical render.
To match this:

1. Select the body mesh
2. Go to **Properties panel → Material Properties** (sphere icon)
3. Find the skin material node tree
4. Adjust these values:
   ```
   Roughness:        0.75 → 0.85  (more matte, less glossy)
   Specular:         0.3  → 0.15  (reduce shine)
   Subsurface:       keep at ~0.01 (subtle skin translucency)
   Subsurface Color: keep warm pink
   ```

> **For SceneKit export:** The procedural Blender skin material won't
> transfer perfectly to SceneKit's Phong model. What DOES transfer:
> - Diffuse color texture (the skin color map) ✓
> - Normal map (surface detail) ✓ (via glTF pipeline)
> - Basic roughness ✓
> 
> This is sufficient — SceneKit's rendering makes it look clean and clinical,
> which is exactly what we want for a medical app.

### 3c. If No Skin Packs Available (Fallback)

If you couldn't install asset packs, create a manual skin material:

1. Select the body mesh
2. Properties → Material → **New**
3. Name it "DermaFusion_Skin"
4. Set Principled BSDF values:
   ```
   Base Color:    Hex #D4B5A0  (warm flesh tone matching reference)
   Roughness:     0.80         (matte medical look)
   Specular:      0.15         (minimal shine)
   Subsurface:    0.01
   Subsurface Color: #E8A090   (pinkish undertone)
   ```

This gives a clean, clinical skin appearance similar to the reference.

---

## Step 4: Remove Extras, Clean Mesh

The reference image is a **bare body only** — no eyes, hair, teeth, etc.

### 4a. Delete Non-Body Parts

MakeHuman/MPFB2 may generate separate objects for eyes, eyebrows,
eyelashes, teeth, tongue, and hair. Delete them all:

1. In the **Outliner** (top-right panel), expand the collection
2. You'll see objects like:
   - `Human` (the body mesh — KEEP THIS)
   - `High-poly-eyes` or similar — DELETE
   - `Eyebrows` — DELETE
   - `Eyelashes` — DELETE
   - `Teeth` — DELETE
   - `Tongue` — DELETE
   - Any hair objects — DELETE

3. Select each non-body object → press **X** → Delete

> **Keep only the body mesh.** The body mesh already has eye socket
> geometry, brow ridges, lips, and ear shapes built into the topology.
> Removing the eye/teeth objects leaves clean facial features.

### 4b. Join If Multiple Body Parts

If the body came as multiple mesh objects (unlikely with MPFB2 "from scratch"):

1. Select all body mesh parts
2. **Ctrl+J** to join into a single object
3. Rename to "Human" in the Outliner

### 4c. Apply All Transforms

1. Select the body mesh
2. **Ctrl+A → All Transforms**

### 4d. Set Smooth Shading

1. Right-click the body mesh → **Shade Smooth**
2. This matches the smooth surface look in the reference

### 4e. Verify A-Pose

The reference shows arms at roughly 30° from the body (A-pose).
MPFB2's default is T-pose. To switch:

1. If you added a rig: MPFB sidebar → **Rigging → Load Pose** → 
   select "A-pose" or manually rotate upper arm bones ~30° down
2. If no rig: the default T-pose is fine — the segmentation script
   handles both T-pose and A-pose. A-pose just looks better in the app.

**Simpler approach:** Skip the rig entirely. Export in T-pose. The arms
will be further out but segmentation works the same. In SceneKit the
user rotates the model anyway, so pose is purely aesthetic.

---

## Step 5: Run Segmentation Script

With the clean body mesh selected:

1. Open **Scripting** workspace (top tab bar)
2. Click **New** → paste contents of `dermafusion_segment_body.py`
3. Set your export path at the top of the script:
   ```python
   EXPORT_PATH = os.path.expanduser("~/Desktop/BodyModel.dae")
   ```
4. Click **Run Script** (Alt+P)

The script:
- Analyzes the mesh bounding box
- Classifies every face into 1 of 12 HAM10000 regions
- Separates into 12 named objects (preserving UV, materials, normals)
- Adds camera + 3-point lighting
- Exports .dae, .obj, and .glb

### Expected Output

```
============================================================
DermaFusion Body Segmenter
============================================================
Input mesh: 'Human' (25834 faces, 13377 verts)
  Bounding box: 1.628 x 1.733 x 0.292
  Up axis: Y, Right axis: X, Forward axis: Z

  ✓ region_scalp              1245 verts  2103 faces  mat:✓  uv:✓
  ✓ region_face                876 verts  1534 faces  mat:✓  uv:✓
  ✓ region_ear                 342 verts   584 faces  mat:✓  uv:✓
  ✓ region_neck                456 verts   812 faces  mat:✓  uv:✓
  ✓ region_chest              2012 verts  3847 faces  mat:✓  uv:✓
  ✓ region_abdomen            1534 verts  2847 faces  mat:✓  uv:✓
  ✓ region_back               2245 verts  4521 faces  mat:✓  uv:✓
  ✓ region_upper_extremity    1876 verts  3423 faces  mat:✓  uv:✓
  ✓ region_lower_extremity    1834 verts  3512 faces  mat:✓  uv:✓
  ✓ region_hand                534 verts   987 faces  mat:✓  uv:✓
  ✓ region_foot                312 verts   534 faces  mat:✓  uv:✓
  ✓ region_genital             111 verts   198 faces  mat:✓  uv:✓

  ✓ ALL 12 REGIONS PRESENT — Ready for SceneKit
```

Note: `mat:✓` confirms the skin material was preserved on each region,
and `uv:✓` confirms UV texture coordinates survived the split.

---

## Step 6: Export for SceneKit

### Recommended Pipeline: glTF → USDZ → .scn

This preserves skin textures and materials with highest fidelity.

#### 6a. Export glTF from Blender

The segmentation script already exports a .glb file. If you need to
re-export manually:

1. Select all `region_*` objects (type "region" in search)
2. Also select CameraNode, DirectionalLightNode, FillLightNode
3. **File → Export → glTF 2.0 (.glb/.gltf)**
4. Settings:
   ```
   Format:           glTF Binary (.glb)
   Include:          ✓ Selected Objects
   Transform:        +Y Up (default)
   Geometry:         ✓ Apply Modifiers
                     ✓ UVs
                     ✓ Normals
                     ✓ Vertex Colors (if any)
   Material:         Export
   Images:           Automatic (embeds textures in .glb)
   ```
5. Save as `BodyModel.glb`

#### 6b. Convert to USDZ (Mac)

1. Open **Reality Converter** (download from Apple Developer site)
2. File → Import → select `BodyModel.glb`
3. Verify:
   - Model renders with skin material (not black/white)
   - All 12 regions are visible as separate objects in the sidebar
   - Rotate to check all sides look correct
4. File → Export → save as `BodyModel.usdz`

#### 6c. Convert to .scn in Xcode

1. Drag `BodyModel.usdz` into Xcode project → `Resources/`
2. Select it in the project navigator
3. Xcode opens the **Scene Editor** with 3D preview
4. Menu: **Editor → Convert to SceneKit Scene File Format (.scn)**
5. Save as `BodyModel.scn` in the same location
6. Verify in Scene Editor:
   - Expand scene graph (bottom-left)
   - Confirm all 12 `region_*` nodes appear
   - Click each node to verify it highlights the correct body area
   - Check that skin material/texture is visible (not wireframe)
7. Delete the `.usdz` from the project (optional, saves bundle size)

---

## Step 7: SceneKit Material Adjustment

After importing, the skin material may need minor tweaking in Xcode's
Scene Editor to match the reference look:

### In Xcode Scene Editor:

1. Select any `region_*` node
2. Open **Material Inspector** (right panel)
3. Adjust to match the matte clinical look:
   ```
   Diffuse:     The skin texture map (should auto-import)
                If no texture: set to color #D4B5A0
   Specular:    #333333 (subtle, not shiny)
   Shininess:   0.15    (very matte)
   Normal:      The normal map (if it imported)
   Ambient:     #282828
   Transparency: 1.0    (fully opaque)
   ```

### Or Adjust at Runtime in Swift:

```swift
/// Configure skin material for the medical mannequin look
func configureSkinMaterial(for node: SCNNode) {
    guard let material = node.geometry?.firstMaterial else { return }
    
    // Matte skin look (matches reference image)
    material.lightingModel = .physicallyBased
    material.diffuse.contents = UIColor(red: 0.82, green: 0.71, blue: 0.60, alpha: 1.0)
    material.roughness.contents = NSNumber(value: 0.85)  // Very matte
    material.metalness.contents = NSNumber(value: 0.0)   // Non-metallic skin
    
    // If you have a skin texture exported from Blender:
    // material.diffuse.contents = UIImage(named: "skin_diffuse")
    // material.normal.contents = UIImage(named: "skin_normal")
}

/// Apply skin material to all body regions at scene load
func configureAllRegions(in scene: SCNScene) {
    let regionNames = [
        "region_scalp", "region_face", "region_ear", "region_neck",
        "region_chest", "region_abdomen", "region_back",
        "region_upper_extremity", "region_lower_extremity",
        "region_hand", "region_foot", "region_genital"
    ]
    for name in regionNames {
        if let node = scene.rootNode.childNode(withName: name, recursively: true) {
            configureSkinMaterial(for: node)
        }
    }
}
```

---

## Step 8: Verify the Final Result

### What It Should Look Like

In the iOS app (or Xcode's Scene Editor), you should see:

✓ Realistic human body with visible facial features (brow, nose, lips)
✓ Warm matte skin tone (not glossy, not gray)  
✓ Clean mesh topology visible at edges (anatomical contour lines)
✓ 12 distinct tappable regions that highlight on selection
✓ Smooth orbital rotation (drag to spin)
✓ Clinical, professional appearance suitable for a medical context

### Visual Comparison Checklist

| Feature | Reference Image | Your Model |
|---------|----------------|------------|
| Skin material | Warm flesh tone, matte | Should match via MPFB skin |
| Facial features | Eyes, nose, mouth, brow | Built into MakeHuman topology |
| Muscle definition | Athletic, visible abs | Muscle 0.6, Weight 0.3 |
| Mesh quality | Clean edge loops | MakeHuman base mesh |
| Pose | A-pose, palms forward | Default or manually posed |
| Hair | Bald | Delete hair objects |
| Accessories | None | Delete eyes, teeth, etc. |

---

## Troubleshooting

### "Model looks too smooth / no muscle definition"
→ Increase Muscle slider to 0.7-0.8 in MPFB's Model panel

### "Skin is too shiny in SceneKit"  
→ Increase roughness to 0.90, decrease specular

### "Facial features are too flat"
→ In MPFB Model → Face panel, increase Brow Ridge, Nose Bridge, 
  Lip Thickness values

### "Mesh topology not visible like the reference"
→ The reference shows edge loops because of the render style.
  In SceneKit with smooth shading, edges blend away — this is correct
  for a medical app. If you WANT visible edges, set:
  ```swift
  material.fillMode = .lines  // Wireframe overlay
  ```
  But for production, smooth shading looks more professional.

### "Export is missing skin texture"
→ Use the glTF (.glb) pipeline — it embeds textures. COLLADA (.dae) 
  sometimes loses texture references. If all else fails, note the texture 
  file path from Blender (Image Editor → image name) and manually add it 
  to your Xcode project, then assign it in the Scene Editor.

### "Arms too wide for the screen"
→ T-pose arms extend far. Either: (a) use A-pose, or (b) adjust the 
  SceneKit camera's field of view to fit, or (c) the user rotates anyway.

---

## File Size Expectations

| Export Format | Approximate Size | Notes |
|---------------|-----------------|-------|
| .glb (with textures) | 2-5 MB | Textures embedded |
| .glb (color only) | 300-500 KB | No texture maps |
| .usdz | 2-5 MB | After Reality Converter |
| .scn | 1-3 MB | After Xcode conversion |
| .dae | 500 KB - 1 MB | Without embedded textures |

For a mobile app, the color-only version (no texture maps, just material
color) at ~500 KB is ideal. The user only needs to see region boundaries,
not skin pore detail.

---

## Summary: 30-Minute Quick Path

If you just want to get this done fast:

1. Install Blender + MPFB2 + System Assets pack (~10 min)
2. Create Human: Gender 1.0, Muscle 0.6, Weight 0.3, Caucasian 1.0 (~2 min)
3. Load skin from library (~1 min)
4. Delete eyes/teeth/hair objects (~2 min)
5. Apply transforms, Shade Smooth (~1 min)
6. Run `dermafusion_segment_body.py` (~2 min)
7. Export .glb → Reality Converter → .usdz → Xcode → .scn (~10 min)
8. Verify 12 nodes in Scene Editor (~2 min)

**Total: ~30 minutes** to a production-quality 3D body model matching
the reference image, with 12 named tappable regions for SceneKit.
