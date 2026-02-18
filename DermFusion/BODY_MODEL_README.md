# DermaFusion 3D Body Model

## Files

| File | Size | Purpose |
|------|------|---------|
| `BodyModel.dae` | 287 KB | **Primary** — COLLADA format, loads natively in SceneKit |
| `BodyModel.obj` | 317 KB | Backup — Wavefront OBJ format (also loads in SceneKit) |
| `BodyModel.mtl` | 1.4 KB | Material definitions for .obj (paired file) |

## Model Specifications

- **Total geometry:** 2,884 vertices / 4,588 triangles (lightweight)
- **Scale:** 1 unit = 1 meter (realistic human: 1.73m tall)
- **Orientation:** Y_UP, facing +Z direction
- **Style:** Gender-neutral medical mannequin (no facial features, no fingers/toes)
- **Materials:** Uniform matte gray (Phong shading), designed for runtime highlighting
- **Scene includes:** Camera (framing full body), 3-point lighting (ambient + key + fill)

## 12 Named Region Nodes

Each region is a **separate SCNNode** for per-region hit testing:

```
region_scalp              → Top of head (dome)
region_face               → Front face area
region_ear                → Both ears (left + right as one node)
region_neck               → Neck cylinder
region_chest              → Front upper torso (above waist)
region_abdomen            → Front lower torso (waist to hips)
region_back               → Full back surface (shoulders to hips)
region_upper_extremity    → Both arms (shoulder to wrist)
region_lower_extremity    → Both legs (hip to ankle)
region_hand               → Both hands
region_foot               → Both feet
region_genital            → Discrete lower front area
```

These names match the HAM10000 `localization` field and the `BodyRegion` enum in the app.

## Xcode Integration (3 Steps)

### Step 1: Import into Xcode

Drag `BodyModel.dae` into your Xcode project's `Resources/` group.
When prompted, check **"Copy items if needed"** and add to the DermaFusion target.

### Step 2: Convert to .scn

1. Select `BodyModel.dae` in Xcode's project navigator
2. Xcode opens it in the **Scene Editor** (3D preview)
3. Menu bar → **Editor → Convert to SceneKit Scene File Format (.scn)**
4. Save as `BodyModel.scn` in the same location
5. Delete the original `BodyModel.dae` from the project (no longer needed)

### Step 3: Verify Node Names

In the Scene Editor, expand the scene graph (bottom-left panel). You should see:

```
Scene
├── CameraNode
├── AmbientLightNode
├── DirectionalLightNode
├── FillLightNode
├── region_scalp          ← tappable body regions
├── region_face
├── region_ear
├── region_neck
├── region_chest
├── region_abdomen
├── region_back
├── region_upper_extremity
├── region_lower_extremity
├── region_hand
├── region_foot
└── region_genital
```

All 12 `region_*` nodes must appear. If any are missing, the `.dae` import failed — try reimporting.

## Swift Usage

### Loading the Scene

```swift
guard let scene = SCNScene(named: "BodyModel.scn") else {
    fatalError("BodyModel.scn not found in app bundle")
}
```

### Finding a Region Node

```swift
let scalp = scene.rootNode.childNode(withName: "region_scalp", recursively: true)
```

### Hit Testing on Tap

```swift
let hitResults = sceneView.hitTest(tapLocation, options: [
    .searchMode: SCNHitTestSearchMode.closest.rawValue,
    .boundingBoxOnly: false
])

if let hit = hitResults.first,
   let nodeName = hit.node.name,
   nodeName.hasPrefix("region_") {
    // Map to BodyRegion enum
    let regionKey = String(nodeName.dropFirst("region_".count))
    selectedRegion = BodyRegion(rawValue: regionKey)
}
```

### Highlighting a Selected Region

```swift
func highlight(node: SCNNode) {
    // Animate to brand primary color at 35% opacity
    let highlightColor = UIColor(
        red: 0.27, green: 0.52, blue: 0.96, alpha: 0.85  // brandPrimary
    )
    SCNTransaction.begin()
    SCNTransaction.animationDuration = 0.25
    node.geometry?.firstMaterial?.diffuse.contents = highlightColor
    node.geometry?.firstMaterial?.emission.contents = UIColor(
        red: 0.27, green: 0.52, blue: 0.96, alpha: 0.15
    )
    SCNTransaction.commit()
}

func resetHighlight(node: SCNNode) {
    let defaultColor = UIColor(red: 0.78, green: 0.78, blue: 0.80, alpha: 1.0)
    SCNTransaction.begin()
    SCNTransaction.animationDuration = 0.2
    node.geometry?.firstMaterial?.diffuse.contents = defaultColor
    node.geometry?.firstMaterial?.emission.contents = UIColor.black
    SCNTransaction.commit()
}
```

## Alternative: Load .obj Directly (No Conversion)

If you prefer skipping the .scn conversion, SceneKit loads `.obj` files too:

1. Drag **both** `BodyModel.obj` and `BodyModel.mtl` into `Resources/`
2. Load with: `SCNScene(named: "BodyModel.obj")`

The `.obj` groups become `SCNNode` names identically to the `.dae` nodes.
The `.scn` conversion is recommended for faster load times in the app.

## Customization

### Adjusting in Xcode Scene Editor

After converting to `.scn`, you can tweak directly in Xcode:
- **Camera position:** Select CameraNode, adjust transform in Inspector
- **Lighting:** Select light nodes, adjust intensity/color/direction
- **Materials:** Select any region node → Material Inspector → change diffuse color
- **Background:** Set scene background to clear (for transparent overlay on SwiftUI)

### Model Modifications

To modify the geometry (e.g., adjust proportions, add detail):
1. Edit the Python generator script (`generate_body_dae.py`)
2. Adjust coordinates in `build_body_regions()`
3. Re-run → re-import → re-convert to `.scn`
