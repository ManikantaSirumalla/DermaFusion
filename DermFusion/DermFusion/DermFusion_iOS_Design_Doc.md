# DermaFusion iOS App — Design Document
## On-Device Multi-Modal Skin Cancer Detection

**Status:** Planning (build after core ML pipeline is complete — post Step 11)
**Target:** iOS 17+, iPhone 12 and later (Neural Engine required for CoreML performance)
**Stack:** SwiftUI, CoreML, Vision, Swift Charts

---


# 1. PRODUCT OVERVIEW

## 1.1 What It Is
A privacy-preserving, on-device skin lesion analysis app that combines dermoscopic/close-up
image capture with clinical metadata to classify skin lesions across 7 diagnostic categories
using the DermaFusion multi-modal deep learning model.

## 1.2 What It Is NOT
- NOT a medical device (prominently disclaimed)
- NOT a replacement for dermatologist consultation
- NOT intended for self-diagnosis
- NOT collecting or transmitting any user health data

## 1.3 Why Build It
- Demonstrates end-to-end deployment (training → optimization → mobile inference)
- Privacy-by-design: all inference on-device, zero data leaves the phone
- Portfolio differentiator: transforms an academic project into a shipped product
- Aligns with real clinical need: triage tool for areas with limited dermatologist access
- Shows CoreML/ONNX model conversion competency (industry-relevant skill)

## 1.4 Target Users
- Primary: Capstone reviewers, hiring managers, journal reviewers (demo/showcase)
- Secondary: Medical students learning dermoscopy, dermatology residents
- NOT: General public for self-diagnosis (this requires regulatory approval)


---


# 2. FEATURE SPECIFICATION — TAB-BY-TAB BREAKDOWN

Navigation: Bottom tab bar with 4 tabs. Tab bar uses SF Symbols, tinted with
DFColors.brandPrimary when active, DFColors.textTertiary when inactive.

```
Tab Bar: [ Scan | History | Learn | About ]
             ★       ↺       📖       ℹ
```


## TAB 1: SCAN (Primary — where users spend 90% of their time)

The entire capture-to-results pipeline is a linear navigation stack within this tab.
Users flow forward through 5 screens, with back navigation available at each step.

### Screen 1.1 — Home / Landing (ScanHomeView)
Purpose: Entry point for new scans. Clean, purposeful, not cluttered.

Layout:
- App name "DermaFusion" centered near top with subtle medical cross icon
- Large hero "New Scan" primary button (DFButton, full-width, brand color)
- Below it: "Choose from Library" secondary button (DFButton, outlined style)
- If user has saved scans: a "Last Scan" card (DFCard) showing:
  - Thumbnail of most recent scan image (small, rounded)
  - Primary diagnosis + confidence ("Melanocytic Nevi — 87%")
  - Risk badge (small DFRiskBadge)
  - Relative timestamp ("2 days ago")
  - Tapping the card navigates to History tab → scan detail
- If no saved scans: omit the last scan card entirely (clean first-time experience)

Edge cases:
- Camera unavailable (Simulator, no camera hardware) → hide "New Scan", show only library picker
- First ever launch → no last scan card, just the two buttons centered vertically

### Screen 1.2 — Image Capture / Selection
Branched: tapping "New Scan" opens camera, tapping "Choose from Library" opens picker.

**Camera path (CameraCaptureView):**
- Full-screen AVFoundation camera preview
- Circular overlay guide (semi-transparent dark border with clear circle in center)
  - Circle diameter: ~70% of screen width
  - Subtle pulsing animation to guide framing
  - Text below circle: "Position the lesion within the circle"
- Capture button at bottom center (DFIcons.capture, 72pt, white circle with border)
- Flash toggle top-left, switch camera top-right (front camera unlikely useful, but include)
- Cancel button (X) top-right to dismiss back to home

**Library path:**
- PHPickerViewController presented as sheet
- Filter: images only (no video)
- Single selection mode

**Both paths lead to → Image Review (ImageReviewView):**
- Full-screen display of captured/selected image
- Pinch-to-zoom and pan to inspect image quality
- "Retake" button (left, secondary style) → returns to camera/picker
- "Use This Image" button (right, primary style) → proceeds to metadata input
- Subtle quality hints: if image is very small (< 200×200), show warning banner:
  "Low resolution image — results may be less accurate"

Edge cases:
- Camera permission not yet requested → system prompt appears, handle both outcomes
- Camera permission denied → show DFBanner with "Camera access needed" + settings link
- Photo library permission denied → same pattern with library-specific message
- Photo library empty → PHPicker handles this natively (shows empty state)
- Image corrupt or unreadable → show error alert, return to home
- Image has extreme aspect ratio (panorama) → still works, center crop handles it
- User rapidly taps capture button → debounce, only process the first tap
- User leaves app mid-capture → release camera session, resume gracefully on return

### Screen 1.3 — Metadata Input (MetadataInputView)
Purpose: Collect clinical metadata that feeds the multi-modal model.

Layout (single scrollable screen):
- Section header: "Patient Information"
- **Age:** horizontal DFSlider (0–100), numeric readout on right side
  - Default: slider at 0 position (user must actively set it)
  - If slider is at 0: show subtle prompt "Tap to set age, or leave blank to skip"
  - "Skip" option: toggle that sets age to nil (model uses age_missing flag)
  - When skipped, slider grays out and shows "Age not provided"
- **Sex:** DFSegmentedControl with 3 options: "Male" | "Female" | "Prefer not to say"
  - Default: "Prefer not to say" (encodes as unknown index 2)
- **Lesion Location:** a tappable row (DFCard style) showing:
  - Body map icon (DFIcons.bodyMap) on left
  - "Select Location" label (or selected region name if already chosen)
  - Chevron on right
  - Tapping PUSHES to Screen 1.3a — Body Map (separate screen)
  - When a region is selected and user returns, this row updates to show:
    - Region name in bold ("Lower Extremity")
    - Small highlighted body silhouette thumbnail
    - Checkmark icon

- **"Analyze" button** at bottom (full-width primary DFButton)
  - DISABLED (grayed out) until body region is selected
  - Tooltip on disabled state: "Select a lesion location to continue"
  - When enabled and tapped → navigate to Screen 1.4 (Analysis)

Edge cases:
- Age 0 intentionally selected (infant) → valid, no warning needed
- Age > 100 → slider caps at 100, no crash
- User skips age AND sex → valid (model handles missing data), but location required
- User changes metadata after seeing results and navigating back → invalidate results
- User taps "Analyze" rapidly → debounce, process only once

### Screen 1.3a — Body Map (BodyMapView) — SEPARATE PUSHED SCREEN
Purpose: Interactive anatomical region selection.

Layout:
- Navigation title: "Select Location"
- "Front / Back" segmented toggle at top (DFSegmentedControl)
- Full-height body silhouette centered on screen
  - SVG-based, scalable, works on all screen sizes
  - Use a human-like neutral mannequin (not robotic): soft anatomical proportions, rounded shoulders/limbs, and no mechanical joints/details
  - Keep the figure non-sexualized and clinically neutral to support medical context
  - Light outline on DFColors.backgroundPrimary
  - Visual spec checklist:
    - Pose: upright front/back standing pose, arms slightly away from torso so side body regions remain tappable
    - Surface style: flat vector with subtle shading only (no musculature, no facial features, no hair details)
    - Tone variants: support at least 3 neutral skin-tone themes for inclusive UI previews (runtime can default to theme token)
    - Stroke and regions: 1.5-2pt outline, clear region boundaries, and minimum 44pt effective touch target per tappable area
    - Selection state: selected region fill at 35% brandPrimary plus 2pt high-contrast border and check indicator near region label
- 12 tappable regions mapped to HAM10000 localization categories:
  - Front: scalp, face, ear, neck, chest, abdomen, upper extremity, lower extremity, hand, foot
  - Back: back (+ shared regions visible from both views: scalp, ear, neck, upper/lower extremity, hand, foot)
  - Genital: small discrete region on front view
- Tap behavior:
  - Tapped region fills with DFColors.bodyMapHighlight (brandPrimary at 35% opacity)
  - Region name label appears near the tapped area or at top of screen
  - Only ONE region selected at a time (tapping a new one deselects the previous)
- "Confirm" button at bottom (primary DFButton) — pops back to metadata input
  - Disabled until a region is tapped
  - Shows selected region name: "Confirm: Lower Extremity"

Edge cases:
- User taps between two regions (ambiguous) → favor the region whose center is closest
- User taps outside all regions → no selection, no crash
- Very small screen (iPhone SE) → regions may be tight; ensure minimum 44pt touch targets
- VoiceOver: each region is a button with label "Select [region name]"

### Screen 1.4 — Analysis Loading (AnalysisView)
Purpose: Visual transition while CoreML inference runs.

Layout:
- Centered DFLoadingIndicator (branded animation — subtle pulse or circular progress)
- "Analyzing..." text below (DFTypography.bodyRegular)
- Small captured image thumbnail above the spinner as context
- Minimum display time: 1.2 seconds (even if inference finishes in 50ms)
  - This prevents a jarring flash and gives the user confidence something meaningful happened
- Auto-navigates to Results screen when both inference completes AND minimum time has elapsed

Edge cases:
- Inference fails → navigate to error state, NOT results (show alert with DFError.inferenceFailed)
- User taps back during loading → cancel inference Task, return to metadata screen
- Inference takes > 5 seconds → show additional message: "Taking longer than usual..."
- App goes to background during inference → inference continues, results ready on return

### Screen 1.5 — Results (ResultsView)
Purpose: Present classification results with clinical context and safety disclaimers.

Layout (scrollable, top to bottom):
- **Hero image section:**
  - Captured image, full-width, rounded corners (DFSpacing.cornerRadius)
  - GradCAM toggle button overlaid in bottom-right corner of image
    - SF Symbol: DFIcons.gradcam
    - Tap: cross-dissolve blend of GradCAM heatmap overlay onto the image
    - Tap again: hide heatmap, show original
    - If GradCAM failed to generate: hide this button entirely (non-fatal failure)

- **Primary result card (DFCard):**
  - Top row: DFRiskBadge (pill shape, icon + text, colored by risk level)
    - Green + shield: "Low Risk"
    - Yellow + triangle: "Moderate Risk"
    - Red + octagon: "High Risk — Consult a Dermatologist"
  - Primary diagnosis: large text (DFTypography.displayMedium)
    e.g., "Melanocytic Nevi"
  - Confidence: large number (DFTypography.displayLarge)
    e.g., "87.3%"
  - "Learn More" text button → navigates to Learn tab's LesionTypeDetailView for this category

- **Probability chart (ProbabilityChartView):**
  - Section header: "All Classifications" (DFTypography.subheadline)
  - Horizontal bar chart using Swift Charts
  - 7 bars, sorted by probability (highest at top)
  - Each bar: colored by DFColors.chartColor(for: category), labeled with category name + percentage
  - Category names use full display names, not abbreviations
  - VoiceOver: reads each bar as "Melanocytic Nevi, 87 percent"

- **Metadata summary:**
  - Small section showing the inputs used: "Age: 55, Sex: Male, Location: Lower Extremity"
  - Styled as muted text (DFTypography.caption, DFColors.textSecondary)

- **Action buttons (horizontal row):**
  - "Save Scan" (DFButton, primary) → saves to SwiftData, shows success toast
    - After saving, button changes to "Saved ✓" (disabled, green tint)
  - "Export PDF" (DFButton, secondary) → generates PDF, opens iOS share sheet
    - PDF includes: image, GradCAM (if available), all probabilities, metadata, disclaimer

- **Disclaimer banner (DFBanner, .disclaimer style):**
  - Pinned at bottom of scroll content (not floating — scrolls with content)
  - DFColors.disclaimerBackground background
  - DFIcons.disclaimer icon + text: "Research tool only — not a medical diagnosis. Always consult a dermatologist."
  - NOT dismissible — always visible

Edge cases:
- GradCAM unavailable → hide toggle button, show results normally
- All probabilities nearly equal (max < 0.20) → show additional DFBanner:
  "Low confidence — the model is uncertain about this image"
- Melanoma very high confidence (> 0.80) → risk badge is red, but language stays hedged
  ("Classification suggests melanoma"), never "You have melanoma"
- User saves scan, then navigates back and changes metadata → saved scan is separate,
  re-analysis creates a new result
- User hasn't saved and hits back → confirmation alert: "Discard this result?"
- Export PDF while GradCAM is still generating → export without GradCAM, note in PDF
- Disk full when saving → catch error, show DFError.storageFull alert


## TAB 2: HISTORY (Scan Records)

MVP scope: chronological list + detail view.
Post-MVP: compare mode, timeline strip, filtering by body region/risk level.

### Screen 2.1 — Scan List (HistoryListView)
Purpose: Browse all saved scans chronologically.

Layout:
- Navigation title: "History"
- List of saved scans (newest first), each row contains:
  - Thumbnail image (60×60pt, rounded, left side)
  - Primary diagnosis name (DFTypography.bodyBold)
  - Confidence percentage (DFTypography.mono, DFColors.textSecondary)
  - Small DFRiskBadge (compact size — just the colored dot + "Low"/"Mod"/"High")
  - Body region name (DFTypography.caption, DFColors.textSecondary)
  - Relative timestamp: "2 hours ago", "Yesterday", "Jan 15, 2026"
  - Chevron on right (standard iOS list disclosure)
- Swipe-to-delete on each row:
  - Red destructive action: "Delete"
  - Confirmation alert: "Delete this scan? This cannot be undone."
- Toolbar: "Delete All" button (destructive, with double-confirmation:
  "Delete all scans? This will permanently erase all saved data.")

**Empty state (DFEmptyState):**
- Shown when zero scans are saved
- Illustration: simple line drawing of a camera/scan icon
- Title: "No Scans Yet"
- Subtitle: "Your saved analyses will appear here"
- CTA button: "Start Your First Scan" → switches to Scan tab

Edge cases:
- Hundreds of scans → lazy loading (List handles this natively), thumbnails only in list
- Corrupted scan record (image data nil) → skip in list, don't crash, log error
- User deletes scan while viewing it in detail → pop back to list
- All scans deleted → transition to empty state with animation

### Screen 2.2 — Scan Detail (ScanDetailView)
Purpose: Full detail view of a single saved scan. Mirrors the Results screen layout.

Layout (reuses most ResultsView components):
- Hero image with GradCAM toggle (if GradCAM data was saved)
- Risk badge + primary diagnosis + confidence
- Probability chart (all 7 classes)
- Metadata summary (age, sex, location used for this scan)
- Timestamp: "Scanned on January 15, 2026 at 3:42 PM"
- User notes section:
  - If notes exist: display with "Edit" button
  - If no notes: "Add a note..." tappable text field
  - Notes are free-text, stored with the scan record
- Action buttons:
  - "Export PDF" (same as results screen)
  - "Delete Scan" (destructive, in toolbar, with confirmation)
- Disclaimer banner (same persistent banner as results screen)

Edge cases:
- GradCAM data wasn't saved with this scan (older scan) → hide toggle, show image only
- Scan from before a model update (different class order) → version the saved probabilities
- Editing notes triggers SwiftData save → handle save failure gracefully


## TAB 3: LEARN (Educational Content)

MVP scope: ABCDE Rule, Lesion Types Guide, Dermoscopy Basics.

### Screen 3.1 — Learn Hub (LearnView)
Purpose: Educational content hub with three sections.

Layout (scrollable):
- Navigation title: "Learn"

- **Section 1: "The ABCDE Rule" card (DFCard, full-width)**
  - Icon: "A B C D E" styled as a horizontal letter strip with each letter in a colored circle
  - Title: "The ABCDE Rule" (DFTypography.subheadline)
  - Subtitle: "Five warning signs to watch for" (DFTypography.caption)
  - Tapping → pushes Screen 3.2 (ABCDERuleView)

- **Section 2: "Lesion Types" header + 7 cards in a 2-column grid**
  - Section header: "Skin Lesion Types" (DFTypography.headline)
  - Each card (DFCard, half-width):
    - Colored left border matching DFColors.chartColor(for: category)
    - Lesion name (DFTypography.bodyBold): "Melanoma"
    - One-line description (DFTypography.caption): "Malignant — most dangerous"
    - Severity indicator: small text — "Malignant", "Pre-cancerous", or "Benign"
      colored appropriately (red, yellow, green)
  - Cards ordered by clinical severity: melanoma first, vascular last
  - Tapping any card → pushes Screen 3.3 (LesionTypeDetailView) for that category

- **Section 3: "Dermoscopy Basics" card (DFCard, full-width)**
  - Icon: DFIcons.education
  - Title: "What is Dermoscopy?" (DFTypography.subheadline)
  - Subtitle: "Understanding dermoscopic imaging" (DFTypography.caption)
  - Tapping → pushes Screen 3.4 (DermoscopyBasicsView)

### Screen 3.2 — ABCDE Rule (ABCDERuleView)
Purpose: Explain the clinical ABCDE mnemonic for melanoma self-screening.

Layout (scrollable):
- Navigation title: "ABCDE Rule"
- Intro paragraph: "The ABCDE rule is a widely used guide to help identify warning
  signs in skin lesions. If a lesion shows any of these characteristics, consult
  a dermatologist." (DFTypography.bodyRegular)

- 5 sections, each containing:
  - Letter in a large colored circle (A, B, C, D, E) — left-aligned
  - Word: "Asymmetry", "Border", "Color", "Diameter", "Evolution" (DFTypography.subheadline)
  - 2-3 sentence plain-language explanation of what to look for
  - Simple illustrative diagram or icon (from app assets, NOT from any copyrighted source)
    Example: For "Asymmetry" — a simple line drawing showing a symmetric vs asymmetric shape

- Footer: disclaimer text reminding users this is educational, not diagnostic

Content source: all text lives in EducationalContent.json, loaded by EducationViewModel.

### Screen 3.3 — Lesion Type Detail (LesionTypeDetailView)
Purpose: Detailed educational information about one lesion category.
This screen is also reachable from "Learn More" on the Results screen.

Layout (scrollable):
- Navigation title: category display name (e.g., "Melanoma")
- Colored header bar using DFColors.chartColor(for: category)

- **Clinical name + abbreviation:** "Melanoma (mel)" (DFTypography.headline)
- **Severity badge:** "Malignant" / "Pre-cancerous" / "Benign" — colored pill
- **Description:** 2-3 sentences in plain language explaining what this lesion type is
  (DFTypography.bodyRegular)
- **Key visual features:** bullet list of 3-5 things to look for
  (e.g., "Irregular borders", "Multiple colors within the lesion", "Diameter > 6mm")
- **Typical demographics:** who is most commonly affected
  (e.g., "More common in adults over 50, especially those with fair skin")
- **Prevalence in dataset:** "This category represents 11.1% of the training data"
  (DFTypography.caption — shows awareness of class imbalance)
- **Example images:** 3-4 images from public-domain or CC-licensed sources
  - Each image has a caption describing what to notice
  - Images stored in app bundle (Resources/Education/)
  - If no example images available for a category: show text-only, no placeholder

- Disclaimer at bottom: "This information is for educational purposes only."

Content source: EducationalContent.json for text, bundled assets for images.

### Screen 3.4 — Dermoscopy Basics (DermoscopyBasicsView)
Purpose: Brief explainer on dermoscopic imaging for non-specialist users.

Layout (scrollable):
- Navigation title: "Dermoscopy Basics"

- **What is dermoscopy?** — 2-3 sentences explaining that dermoscopy uses a special
  magnifying device (dermatoscope) to see structures beneath the skin surface
  that are invisible to the naked eye.

- **How is it different from a regular photo?** — Explains polarized light,
  10x magnification, elimination of surface reflection. Include a simple
  side-by-side comparison diagram (clinical photo vs dermoscopic image of same lesion).

- **Why does AI work well with dermoscopy?** — Brief explanation that standardized
  imaging + consistent lighting + magnified detail makes it well-suited for
  computer vision analysis. References the HAM10000 dataset.

- **What does DermaFusion analyze?** — 1-2 sentences: "DermaFusion analyzes
  dermoscopic images alongside clinical metadata (age, sex, lesion location) using
  a multi-modal deep learning model to classify lesions into 7 categories."

Content source: EducationalContent.json


## TAB 4: ABOUT (Transparency & Trust)

### Screen 4.1 — About (AboutView)
Purpose: App info, model transparency, disclaimer, privacy, credits.

Layout (scrollable, grouped sections with DFCard containers):

- **Section 1: App Identity**
  - App icon (small, centered)
  - "DermaFusion" (DFTypography.headline)
  - Version number: "Version 1.0.0 (Build 1)" (DFTypography.caption)
  - One-line tagline: "On-device multi-modal skin lesion analysis"

- **Section 2: Model Information (DFCard)**
  - Section header: "About the Model" with DFIcons.education
  - **Architecture:** "EfficientNet-B4 image encoder + clinical metadata encoder
    with [fusion type] multi-modal fusion"
  - **Training data:** "Trained on the HAM10000 dataset (10,015 dermoscopic images)
    and ISIC 2019 extensions"
  - **Performance:** key metrics from model card:
    - Balanced accuracy: XX%
    - Macro F1: XX%
    - Melanoma sensitivity: XX% (most important clinically)
  - **Known limitations:**
    - "Optimized for dermoscopic images; clinical/smartphone photos may produce
      less accurate results"
    - "Training data predominantly represents lighter skin tones; performance
      may vary across skin types"
    - "Rare lesion types (dermatofibroma, vascular) have fewer training examples"
  - **What is GradCAM?** — 1-2 sentences: "GradCAM (Gradient-weighted Class Activation
    Mapping) highlights the regions of the image that most influenced the model's
    classification, helping you understand what the AI focused on."

- **Section 3: Medical Disclaimer (DFCard, DFColors.disclaimerBackground)**
  - Full disclaimer text, always visible, not collapsible:
    "DermaFusion is a research and educational tool. It is NOT a medical device
     and has NOT been approved by the FDA or any regulatory authority. Results
     should NOT be used for self-diagnosis. Always consult a qualified
     dermatologist for skin concerns."

- **Section 4: Privacy (DFCard)**
  - Section header: "Your Privacy" with SF Symbol "lock.shield.fill"
  - Text: "All analysis is performed entirely on your device. DermaFusion never
    collects, transmits, or stores your health data on any server. Images and
    results are saved only in the app's local storage. When you delete a scan
    or the app, all data is permanently erased."
  - "App Store Privacy: Data Not Collected" (DFTypography.caption, bold)

- **Section 5: Acknowledgments (DFCard)**
  - Section header: "Acknowledgments"
  - Dataset credits:
    - "HAM10000 — Tschandl et al., Scientific Data, 2018"
    - "ISIC Archive — International Skin Imaging Collaboration"
  - Framework credits: "Built with Apple CoreML, Vision, Swift Charts, and SwiftData"
  - Open source: link to GitHub repository (if public)

- **Section 6: Data Management**
  - "Clear All Scan History" button (destructive style, DFColors.destructive)
    - Requires double confirmation:
      First alert: "Delete all scans? This will permanently erase all saved data."
      Confirm → Second alert: "Are you sure? This cannot be undone."
      Confirm → delete all, show success toast
  - Total storage used: "X scans (XX MB)" (DFTypography.caption)


## CROSS-TAB BEHAVIORS

### First Launch Flow
Before ANY tab is accessible:
1. Full-screen DisclaimerView appears (modal, non-dismissible)
2. Full disclaimer text displayed
3. "I Understand" button at bottom (DFButton, primary)
4. Tapping stores hasAcceptedDisclaimer = true in UserDefaults
5. Dismisses modal, reveals tab bar and Scan tab home screen
6. This screen NEVER appears again unless the app is reinstalled

### Tab Switching Behaviors
- Switching tabs preserves each tab's navigation stack state
- "Learn More" on Results screen → switches to Learn tab + pushes LesionTypeDetailView
  - When user taps back in Learn tab, they return to Learn hub (not Results)
  - User can switch back to Scan tab to find their results intact
- "Start Your First Scan" on History empty state → switches to Scan tab
- Deep linking from last scan card on home → switches to History tab + pushes detail

### Medical Disclaimer Presence
The disclaimer banner (DFBanner) appears on these screens and ONLY these:
- Screen 1.5: Results (persistent, non-dismissible)
- Screen 2.2: Scan Detail (persistent, non-dismissible)
- Screen 3.3: Lesion Type Detail (footer text, lighter)
- Screen 4.1: About (full text in dedicated section)
It does NOT appear on: camera, metadata input, history list, learn hub, ABCDE rule,
dermoscopy basics. Over-disclaimering causes banner blindness.


## POST-MVP ENHANCEMENTS (Build After Core MVP Ships)

### History Enhancements
- Compare mode: side-by-side view of two scans (same or different body regions)
- Timeline strip at top of history list showing dots on dates with scans
- Filter by body region, risk level, or date range
- Search scans by diagnosis name

### Learn Enhancements
- "How the AI Works" section explaining multi-modal fusion, GradCAM internals
- Interactive GradCAM explorer: upload any image and see what the model focuses on
- Quiz mode: show an image, guess the classification, see if model agrees

### Export Enhancements
- Export full scan history as CSV (date, diagnosis, confidence, risk, metadata)
- Share scan comparison as image (side-by-side screenshot)

### Accessibility Enhancements
- Full VoiceOver narration mode for results ("Your scan of the lower extremity,
  analyzed on January 15th, suggests melanocytic nevi with 87% confidence.
  Risk level: low.")
- Haptic feedback for risk level on results reveal


---


# 3. TECHNICAL ARCHITECTURE

## 3.1 System Diagram

```
┌──────────────────────────────────────────────────┐
│                   iOS App Layer                   │
│                                                    │
│  ┌────────────┐  ┌──────────────┐  ┌───────────┐ │
│  │   Camera/   │  │   Metadata   │  │  History   │ │
│  │   Gallery   │  │    Input     │  │  (CoreData)│ │
│  └─────┬──────┘  └──────┬───────┘  └───────────┘ │
│        │                │                          │
│  ┌─────▼────────────────▼───────┐                 │
│  │    Preprocessing Pipeline     │                 │
│  │  - Resize to 380×380         │                 │
│  │  - Shades of Gray            │                 │
│  │  - ImageNet normalization    │                 │
│  │  - Metadata tensor encoding  │                 │
│  └─────────────┬────────────────┘                 │
│                │                                   │
│  ┌─────────────▼────────────────┐                 │
│  │     CoreML Model Engine       │                 │
│  │  ┌─────────────────────────┐ │                 │
│  │  │  DermaFusion.mlpackage  │ │                 │
│  │  │  - Image encoder        │ │                 │
│  │  │  - Metadata encoder     │ │                 │
│  │  │  - Fusion module        │ │                 │
│  │  │  - Classifier           │ │                 │
│  │  └─────────────────────────┘ │                 │
│  │  Runs on Neural Engine (ANE) │                 │
│  └─────────────┬────────────────┘                 │
│                │                                   │
│  ┌─────────────▼────────────────┐                 │
│  │     Results Processing        │                 │
│  │  - Softmax → probabilities   │                 │
│  │  - Risk level classification │                 │
│  │  - GradCAM generation        │                 │
│  └──────────────────────────────┘                 │
│                                                    │
└──────────────────────────────────────────────────┘
```

## 3.2 Model Conversion Pipeline

### Step 1: Export from PyTorch
```python
# After training is complete (post Step 8)
import torch
import coremltools as ct

# Load best model checkpoint
model = build_model(config)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Trace the model (CoreML needs traced or scripted models)
# Must handle BOTH inputs: image + metadata
dummy_image = torch.randn(1, 3, 380, 380)
dummy_metadata = torch.randn(1, NUM_METADATA_FEATURES)

# Option A: Trace directly (if model supports it)
traced = torch.jit.trace(model, (dummy_image, dummy_metadata))

# Option B: Export to ONNX first, then convert
torch.onnx.export(
    model,
    (dummy_image, dummy_metadata),
    "dermafusion.onnx",
    input_names=["image", "metadata"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "metadata": {0: "batch"}},
)
```

### Step 2: Convert to CoreML
```python
import coremltools as ct

# From traced PyTorch model:
mlmodel = ct.convert(
    traced,
    inputs=[
        ct.ImageType(name="image", shape=(1, 3, 380, 380),
                     scale=1/255.0, bias=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                     color_layout=ct.colorlayout.RGB),
        ct.TensorType(name="metadata", shape=(1, NUM_METADATA_FEATURES)),
    ],
    outputs=[
        ct.TensorType(name="logits"),
    ],
    minimum_deployment_target=ct.target.iOS17,
    compute_precision=ct.precision.FLOAT16,  # Faster on Neural Engine
)

# Add metadata
mlmodel.author = "DermaFusion"
mlmodel.short_description = "Multi-modal skin lesion classification"
mlmodel.version = "1.0.0"

# Add class labels
mlmodel.user_defined_metadata["classes"] = "MEL,NV,BCC,AKIEC,BKL,DF,VASC"

mlmodel.save("DermaFusion.mlpackage")
```

### Step 3: Validate on device
- Compare CoreML output vs PyTorch output on 100 test images
- Assert max absolute difference < 0.01 for all predictions
- Benchmark inference time on target device

### GradCAM on CoreML
Two options:
- **Option A (recommended):** Compute GradCAM in Python, export as a separate utility
  model that outputs the heatmap. Two CoreML models: one for classification, one for GradCAM.
- **Option B:** Compute GradCAM natively in Swift using intermediate layer outputs.
  More complex but single model. Requires configuring CoreML to expose intermediate layers.

Recommendation: Start with Option A. Export a second CoreML model that takes the same
inputs but outputs the GradCAM heatmap as an image.


## 3.3 On-Device Preprocessing (Swift)

```swift
// ImagePreprocessor.swift
class ImagePreprocessor {

    /// Apply Shades of Gray color constancy (must match Python implementation)
    func shadesOfGray(_ image: CGImage, power: Int = 6) -> CGImage {
        // 1. Get pixel buffer
        // 2. Compute channel-wise power mean
        // 3. Compute gray = mean of channel means
        // 4. Scale each channel: pixel * (gray / channel_mean)
        // 5. Clip to [0, 255]
        // NOTE: Must produce IDENTICAL output to Python version
        // Validate by comparing outputs on 50 test images
    }

    /// Encode metadata to tensor matching training pipeline
    func encodeMetadata(age: Int?, sex: Sex, location: BodyLocation) -> MLMultiArray {
        // age: normalize by /100.0, set missing flag
        // sex: integer index (male=0, female=1, unknown=2)
        // localization: integer index matching training encoding
        // CRITICAL: use same encoding as training pipeline
    }

    /// Center crop to square then resize
    func preprocessImage(_ image: CGImage, targetSize: Int = 380) -> CVPixelBuffer {
        // 1. Center crop to square
        // 2. Resize to targetSize × targetSize
        // 3. Apply Shades of Gray
        // 4. Return as CVPixelBuffer for CoreML
    }
}
```


## 3.4 Data Persistence (Local Only)

```swift
// Core Data model for scan history
// ALL data stays on device — zero network calls for health data

@Model
class ScanRecord {
    var id: UUID
    var timestamp: Date
    var imageData: Data          // JPEG compressed original image
    var age: Int?
    var sex: String
    var lesionLocation: String
    var predictions: [String: Float]  // class_name: probability
    var primaryDiagnosis: String
    var riskLevel: RiskLevel     // green, yellow, red
    var gradcamImageData: Data?  // GradCAM overlay
    var userNotes: String?
}
```


---


# 4. UI/UX DESIGN

## 4.1 Screen Map (Complete)

```
FIRST LAUNCH (one-time modal):
  └── DisclaimerView → "I Understand" → stores flag → reveals app
        ↓
TAB BAR: [ Scan ★ | History ↺ | Learn 📖 | About ℹ ]

TAB 1 — SCAN (Navigation Stack):
  ScanHomeView
    ├── "New Scan" → CameraCaptureView → ImageReviewView ─┐
    ├── "Choose from Library" → PHPicker → ImageReviewView ─┤
    │                                                        │
    │   ┌────────────────────────────────────────────────────┘
    │   ▼
    │  MetadataInputView (age slider, sex picker, location row)
    │   │
    │   ├── "Select Location" → BodyMapView (pushed, front/back toggle)
    │   │                         └── "Confirm" → pops back with selection
    │   │
    │   └── "Analyze" → AnalysisView (loading, 1.2s min)
    │                       │
    │                       ▼
    │                   ResultsView
    │                     ├── GradCAM toggle (overlay on/off)
    │                     ├── Probability chart (7 classes, Swift Charts)
    │                     ├── Risk badge (green/yellow/red)
    │                     ├── "Save Scan" → SwiftData persist
    │                     ├── "Export PDF" → share sheet
    │                     └── "Learn More" → switches to Learn tab
    │
    └── "Last Scan" card → switches to History tab → ScanDetailView

TAB 2 — HISTORY (Navigation Stack):
  HistoryListView (chronological, swipe-to-delete)
    ├── Empty state → "Start Your First Scan" → switches to Scan tab
    └── Row tap → ScanDetailView
                    ├── GradCAM toggle
                    ├── Probability chart
                    ├── User notes (add/edit)
                    ├── "Export PDF"
                    └── "Delete Scan"

TAB 3 — LEARN (Navigation Stack):
  LearnView (hub with 3 sections)
    ├── "ABCDE Rule" card → ABCDERuleView (5 warning signs)
    ├── Lesion type card → LesionTypeDetailView (per-category education)
    │     (also reachable from "Learn More" on ResultsView)
    └── "Dermoscopy Basics" card → DermoscopyBasicsView

TAB 4 — ABOUT (Single Scrollable Screen):
  AboutView
    ├── App identity (icon, name, version)
    ├── Model information (architecture, training data, metrics, limitations)
    ├── Medical disclaimer (full text, always visible)
    ├── Privacy statement
    ├── Acknowledgments (dataset citations, frameworks)
    └── Data management ("Clear All Scan History")
```

## 4.2 Design Principles

**Medical context = calm, trustworthy aesthetic**
No bright gamified colors. Primary palette is deep medical blue (brand) with
whites, soft grays, and muted teal for secondary actions. The app should feel
like a professional clinical tool, not a consumer health gimmick.

**Risk colors are functional, never decorative**
Green (#34C759), yellow (#FF9500), and red (#FF3B30) appear ONLY in risk
assessment contexts (risk badges, severity indicators). They are never used
for buttons, backgrounds, or decorative elements. Every use of risk color is
paired with an icon and text label for colorblind accessibility.

**Disclaimer is present but not obnoxious**
The medical disclaimer appears on screens that show classification results
(ResultsView, ScanDetailView) and in the About tab. It does NOT appear on
every single screen — over-disclaimering causes banner blindness and is
counterproductive. The disclaimer banner uses a warm neutral background
(not yellow or red) to avoid false urgency.

**Accessibility is foundational, not an afterthought**
All text supports Dynamic Type. All interactive elements have VoiceOver labels.
All touch targets are ≥ 44pt. Risk information is conveyed through icon + text +
color (never color alone). Reduced Motion is respected (animations skip).

**Information architecture is shallow**
Maximum navigation depth from any tab root to deepest screen is 4 taps (Scan tab:
home → camera → metadata → body map). Most flows are 2-3 taps deep. Users should
never feel lost or unable to get back to where they started.


---


# 5. PROJECT STRUCTURE

```
DermaFusion-iOS/
├── DermaFusion.xcodeproj
├── .cursorrules                           # Cursor AI coding standards (see separate file)
├── DermaFusion/
│   ├── App/
│   │   ├── DermaFusionApp.swift           # @main entry point, model preload, SwiftData container
│   │   └── ContentView.swift              # Tab bar root: Scan | History | Learn | About
│   │
│   ├── Core/
│   │   ├── DesignSystem/
│   │   │   ├── DFDesignSystem.swift        # Unified design token hub (namespace)
│   │   │   ├── DFColors.swift              # All colors: brand, surface, text, risk, chart
│   │   │   ├── DFTypography.swift          # All font styles, Dynamic Type scaled
│   │   │   ├── DFSpacing.swift             # 4pt-based spacing scale + semantic spacing
│   │   │   ├── DFShadows.swift             # Card, elevated, modal shadow definitions
│   │   │   └── DFIcons.swift               # SF Symbol constants + custom icon refs
│   │   │
│   │   ├── Components/
│   │   │   ├── DFButton.swift              # Primary, secondary, destructive styles
│   │   │   ├── DFCard.swift                # Surface + shadow + radius container
│   │   │   ├── DFBanner.swift              # Info, warning, disclaimer banner variants
│   │   │   ├── DFRiskBadge.swift           # Green/yellow/red pill (icon + text)
│   │   │   ├── DFLoadingIndicator.swift    # Branded analysis loading animation
│   │   │   ├── DFSegmentedControl.swift    # Styled segmented picker
│   │   │   ├── DFSlider.swift              # Slider with numeric readout
│   │   │   └── DFEmptyState.swift          # Illustration + message + CTA placeholder
│   │   │
│   │   ├── Extensions/
│   │   │   ├── CGImage+Preprocessing.swift  # Shades of Gray, resize, center crop
│   │   │   ├── UIImage+Utilities.swift      # Orientation fix, JPEG compression, thumbnail
│   │   │   ├── Color+Hex.swift              # Init Color from hex string
│   │   │   ├── View+Accessibility.swift     # VoiceOver helper modifiers
│   │   │   ├── Date+Formatting.swift        # Relative and absolute date strings
│   │   │   └── Array+Safe.swift             # Safe subscript (nil instead of crash)
│   │   │
│   │   └── Errors/
│   │       └── DFError.swift                # Unified error enum with user-facing messages
│   │
│   ├── Models/
│   │   ├── DiagnosisResult.swift            # Prediction output: probabilities, risk, metadata
│   │   ├── LesionCategory.swift             # 7-class enum + display names + chart colors
│   │   ├── RiskLevel.swift                  # Enum: low/moderate/high + thresholds
│   │   ├── BodyRegion.swift                 # 12 anatomical regions from HAM10000
│   │   ├── ScanRecord.swift                 # SwiftData @Model for persistence
│   │   ├── MetadataInput.swift              # Struct: age, sex, body region (user input)
│   │   └── Sex.swift                        # Enum: male/female/unspecified + encoding
│   │
│   ├── Services/
│   │   ├── ModelService.swift               # CoreML inference (singleton, async load)
│   │   ├── ImagePreprocessor.swift          # Crop → Shades of Gray → resize pipeline
│   │   ├── GradCAMService.swift             # Heatmap generation + overlay compositing
│   │   ├── PersistenceService.swift         # SwiftData CRUD for ScanRecord
│   │   └── PDFExportService.swift           # PDF report generation with disclaimer
│   │
│   ├── ViewModels/
│   │   ├── ScanViewModel.swift              # Capture → preprocess → infer → result orchestration
│   │   ├── BodyMapViewModel.swift           # Region selection state, front/back toggle
│   │   ├── MetadataInputViewModel.swift     # Age/sex/region validation + encoding
│   │   ├── ResultsViewModel.swift           # Result formatting, GradCAM toggle, save/export
│   │   ├── HistoryViewModel.swift           # Load, delete, empty state for scan records
│   │   └── EducationViewModel.swift         # Load educational content from JSON
│   │
│   ├── Views/
│   │   ├── Launch/
│   │   │   └── DisclaimerView.swift          # First-launch modal, must accept to proceed
│   │   │
│   │   ├── Tab1_Scan/
│   │   │   ├── ScanHomeView.swift            # Landing: New Scan + Library + last scan card
│   │   │   ├── CameraCaptureView.swift       # AVFoundation camera + circular overlay
│   │   │   ├── ImageReviewView.swift         # Pinch-to-zoom review + retake/confirm
│   │   │   ├── MetadataInputView.swift       # Age slider + sex picker + location row
│   │   │   ├── BodyMapView.swift             # Pushed screen: interactive body silhouette
│   │   │   ├── AnalysisView.swift            # Loading animation (1.2s min display)
│   │   │   └── ResultsView.swift             # Hero image + GradCAM + chart + risk + actions
│   │   │
│   │   ├── Tab2_History/
│   │   │   ├── HistoryListView.swift         # Chronological scan list + swipe-to-delete
│   │   │   └── ScanDetailView.swift          # Full detail of saved scan + notes + export
│   │   │
│   │   ├── Tab3_Learn/
│   │   │   ├── LearnView.swift               # Hub: ABCDE card + lesion grid + dermoscopy card
│   │   │   ├── ABCDERuleView.swift           # 5 warning signs with illustrations
│   │   │   ├── LesionTypeDetailView.swift    # Per-category education + example images
│   │   │   └── DermoscopyBasicsView.swift    # What is dermoscopy, how AI uses it
│   │   │
│   │   ├── Tab4_About/
│   │   │   └── AboutView.swift               # Model info, disclaimer, privacy, credits, data mgmt
│   │   │
│   │   └── SharedComponents/
│   │       ├── ProbabilityChartView.swift     # Horizontal bar chart (Swift Charts) — used in Results + Detail
│   │       └── GradCAMOverlayView.swift       # Image with toggleable heatmap — used in Results + Detail
│   │
│   └── Resources/
│       ├── Assets.xcassets                    # App icon, color sets (light+dark), body map images
│       ├── DermaFusion.mlpackage              # CoreML classification model
│       ├── DermaFusionGradCAM.mlpackage       # CoreML GradCAM model
│       ├── EducationalContent.json            # Lesion descriptions, ABCDE content, dermoscopy text
│       └── Education/                          # Example images for lesion type detail screens
│           ├── melanoma/                       # 3-4 CC-licensed example images
│           ├── nv/
│           ├── bcc/
│           ├── akiec/
│           ├── bkl/
│           ├── df/
│           └── vascular/
│
├── DermaFusionTests/
│   ├── Services/
│   │   ├── ImagePreprocessorTests.swift       # Parity vs Python (50 reference images)
│   │   ├── ModelServiceTests.swift            # CoreML vs PyTorch (100 test images)
│   │   └── MetadataEncodingTests.swift        # Encoding matches training pipeline
│   ├── ViewModels/
│   │   ├── ScanViewModelTests.swift           # State transitions, error handling
│   │   ├── HistoryViewModelTests.swift        # CRUD, empty state, delete
│   │   └── ResultsViewModelTests.swift        # Risk calculation, formatting
│   └── Models/
│       ├── RiskLevelTests.swift               # Threshold boundary cases
│       └── LesionCategoryTests.swift          # Display names, model index mapping
│
├── DermaFusionUITests/
│   ├── DisclaimerFlowUITests.swift            # Cannot bypass first-launch disclaimer
│   └── ScanFlowUITests.swift                  # Full capture → metadata → results flow
│
└── Scripts/
    ├── convert_to_coreml.py                   # PyTorch → CoreML (.mlpackage) conversion
    ├── validate_coreml.py                     # Parity: CoreML vs PyTorch on test set
    └── export_gradcam_model.py                # Export GradCAM as separate CoreML model
```


---


# 6. VALIDATION REQUIREMENTS

## 6.1 Model Parity Tests
Before shipping, validate that iOS predictions match Python predictions:

```python
# validate_coreml.py
# Run on 100 test images from HAM10000 test set
for image, metadata, label in test_loader:
    pytorch_logits = pytorch_model(image, metadata)
    coreml_logits = coreml_model.predict({"image": image, "metadata": metadata})

    # Must match within floating point tolerance
    assert np.allclose(pytorch_logits, coreml_logits, atol=0.01), \
        f"Prediction mismatch! Max diff: {max_diff}"

# Report: X/100 images within tolerance
# Any failure = DO NOT SHIP, debug conversion pipeline
```

## 6.2 Preprocessing Parity Tests
```swift
// PreprocessorTests.swift
// Compare Swift Shades of Gray output vs Python output on same images
func testShadesOfGrayParity() {
    // Load 10 reference images + their Python-preprocessed versions
    // Apply Swift preprocessing
    // Assert pixel-wise max absolute difference < 2 (uint8 rounding)
}

func testMetadataEncodingParity() {
    // Encode same (age=55, sex=male, location=back) in Python and Swift
    // Assert identical tensor values
}
```

## 6.3 Performance Benchmarks
- Inference time: < 200ms on iPhone 12 (target: < 100ms)
- App launch to camera ready: < 2 seconds
- Memory footprint: < 200MB during inference
- Model file size: < 50MB (use FLOAT16 quantization)

## 6.4 App Store Considerations
- Medical disclaimer is MANDATORY for App Store approval
- Must NOT claim to diagnose, treat, or prevent any condition
- Category: "Education" or "Health & Fitness" (NOT "Medical")
- Age rating: 12+ (medical/health content)
- Privacy label: "Data Not Collected" (all processing on-device)


---


# 7. IMPLEMENTATION TIMELINE

Build AFTER core ML pipeline is complete (after Step 11 in IMPLEMENTATION_FLOW.md).

### Week 1: Model Conversion
- [ ] Export best PyTorch model to ONNX / TorchScript
- [ ] Convert to CoreML (.mlpackage)
- [ ] Export GradCAM model separately
- [ ] Run parity validation (100 images, <0.01 tolerance)
- [ ] Benchmark inference time on device

### Week 2: Core App Shell
- [ ] Xcode project setup with SwiftUI
- [ ] Tab-based navigation structure
- [ ] Camera capture view (AVFoundation)
- [ ] Photo library picker
- [ ] Image preprocessing pipeline in Swift
- [ ] Validate preprocessing parity vs Python

### Week 3: Inference & Results
- [ ] CoreML inference service (ModelService.swift)
- [ ] Metadata input views (age slider, sex selector)
- [ ] Body map interactive view
- [ ] Results screen with probability chart (Swift Charts)
- [ ] GradCAM overlay toggle
- [ ] Risk level classification logic

### Week 4: Polish & History
- [ ] Scan history with Core Data persistence
- [ ] Educational content screens
- [ ] PDF export functionality
- [ ] Disclaimer system
- [ ] Accessibility pass (VoiceOver, Dynamic Type)
- [ ] App icon and launch screen

### Week 5: Testing & Submission
- [ ] Full test suite (preprocessing parity, inference parity, UI tests)
- [ ] Performance optimization (model quantization if needed)
- [ ] TestFlight beta for review
- [ ] App Store screenshots and metadata
- [ ] Submit for review


---


# 8. FUTURE EXTENSIONS (Not in Scope Now)

These are ideas for post-launch iteration, not MVP requirements:

- **Apple Watch companion:** Quick risk check with photo from wrist camera
- **HealthKit integration:** Store scan results in Health app (with user consent)
- **Multi-language support:** Localize for Spanish, Portuguese, German (major dermatology research countries)
- **Model updates:** OTA model updates via CloudKit (swap .mlpackage without app update)
- **ARKit overlay:** Live camera AR overlay showing GradCAM heatmap in real-time
- **Paired dermoscope support:** Connect to a Bluetooth dermoscope (DermLite, etc.)
- **Federated learning:** On-device fine-tuning with user-consented data (research)
