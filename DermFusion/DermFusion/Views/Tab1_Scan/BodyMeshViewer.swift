//
//  BodyMeshViewer.swift
//  DermFusion
//
//  SceneKit body mesh viewer:
//  • Loads USDZ full-body mesh (male/female via Sex)
//  • Y-axis rotation only (360°); no zoom, pan, or depth shift
//  • Tap hit-tests mesh surface; marker is placed in mesh local space so it stays locked
//  • Returns position, UV, face index, normal, and 18-region classification with Left/Right/Center
//

import SceneKit
import SwiftUI
import UIKit

// MARK: - Tap Result & 18-Region Classification

/// Anatomical side for region label (patient perspective).
enum BodySide: String {
    case left = "Left"
    case right = "Right"
    case center = "Center"
}

/// Internal 18-region enum for spatial classification; maps to app BodyRegion.
private enum BodyPartRegion: String {
    case scalp, face, neck, chest, abdomen
    case upperBack = "Upper Back"
    case midBack = "Mid Back"
    case lowerBack = "Lower Back"
    case glutes = "Glutes"
    case shoulder = "Shoulder"
    case upperArm = "Upper Arm"
    case forearm = "Forearm"
    case hand = "Hand"
    case hip = "Hip"
    case thigh = "Thigh"
    case knee = "Knee"
    case lowerLeg = "Lower Leg"
    case foot = "Foot"
    case unknown = "Unknown"

    var toBodyRegion: BodyRegion {
        switch self {
        case .scalp: return .scalp
        case .face: return .face
        case .neck: return .neck
        case .chest: return .chest
        case .abdomen: return .abdomen
        case .upperBack, .midBack, .lowerBack, .glutes: return .back
        case .shoulder, .upperArm, .forearm: return .upperExtremity
        case .hand: return .hand
        case .hip, .thigh, .knee, .lowerLeg: return .lowerExtremity
        case .foot: return .foot
        case .unknown: return .back
        }
    }
}

/// Tap payload: mesh hit + 18-region label and app BodyRegion.
struct BodyMeshTapResult {
    let worldPosition: SCNVector3
    let localPosition: SCNVector3
    let surfaceNormal: SCNVector3
    let faceIndex: Int
    let textureCoordinates: CGPoint
    let geometryIndex: Int
    let normalizedPosition: SIMD3<Float>
    /// e.g. "Left Hand", "Mid Back"
    let regionDisplayName: String
    /// For app navigation/analysis (HAM10000-style regions).
    let mappedBodyRegion: BodyRegion
    let side: BodySide
    // Legacy names for compatibility
    var localPositionInModel: SCNVector3 { localPosition }
    var localNormalInModel: SCNVector3 { surfaceNormal }
    var nodeName: String? { nil }
    var geometryName: String? { nil }
    var materialName: String? { nil }
}

// MARK: - SwiftUI Representable

struct BodyMeshViewerRepresentable: UIViewControllerRepresentable {
    let sex: Sex
    var onSpotSelected: ((BodyMeshTapResult) -> Void)?
    @Environment(\.colorScheme) private var colorScheme

    func makeUIViewController(context: Context) -> BodyMeshViewerController {
        let vc = BodyMeshViewerController(sex: sex)
        vc.onSpotSelected = onSpotSelected
        return vc
    }

    func updateUIViewController(_ uiViewController: BodyMeshViewerController, context: Context) {
        uiViewController.onSpotSelected = onSpotSelected
        uiViewController.applyBackgroundForColorScheme(colorScheme)
        if uiViewController.currentSex != sex {
            uiViewController.switchSex(to: sex)
        }
    }

    static func hasMesh(for sex: Sex) -> Bool {
        BodyMeshViewerController.resolveScenePath(for: sex) != nil
    }
}

// MARK: - Controller

final class BodyMeshViewerController: UIViewController {

    private(set) var currentSex: Sex
    var onSpotSelected: ((BodyMeshTapResult) -> Void)?

    private var scnView: SCNView!
    private var scene: SCNScene!
    private var cameraNode: SCNNode!
    /// Pivot at origin; we rotate this for Y-axis spin. Hierarchy: root -> pivotNode -> meshNode.
    private var pivotNode: SCNNode!
    /// Mesh geometry node; markers attach here so they rotate with the body.
    private var meshNode: SCNNode!
    private var markerNode: SCNNode?

    private var meshBoundsMin = SCNVector3(0, 0, 0)
    private var meshBoundsMax = SCNVector3(0, 0, 0)
    private var baseCameraDistance: Float = 0

    private var currentAngleY: Float = 0
    private var startAngleY: Float = 0

    init(sex: Sex) {
        self.currentSex = sex
        super.init(nibName: nil, bundle: nil)
    }

    required init?(coder: NSCoder) {
        self.currentSex = .unspecified
        super.init(coder: coder)
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        setupSceneView()
        setupScene()
        setupCamera()
        setupLighting()
        loadBody(for: currentSex)
        setupGestures()
        applyBackgroundForTraitCollection(traitCollection)
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleReset),
            name: .resetBodyView,
            object: nil
        )
    }

    deinit {
        NotificationCenter.default.removeObserver(self, name: .resetBodyView, object: nil)
    }

    @objc private func handleReset() {
        resetToFront()
        clearMarker()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        guard let camera = cameraNode?.camera else { return }
        let aspect = scnView.bounds.width / max(1, scnView.bounds.height)
        let referenceAspect: CGFloat = 9.0 / 19.5
        let baseFOV: CGFloat = 28
        // Wider aspect (e.g. iPad): larger FOV so body fills width; same apparent size as iPhone
        let fov = baseFOV * (aspect / referenceAspect)
        camera.fieldOfView = min(78, max(24, fov))
        // On wide screens (iPad) move camera closer so body appears bigger
        let rawScale = Float(referenceAspect / aspect)
        let distanceScale: Float = aspect > referenceAspect * 1.1 ? max(0.30, min(1, rawScale)) : 1
        let distance = baseCameraDistance * distanceScale
        cameraNode?.position = SCNVector3(0, 0, distance)
    }

    override func traitCollectionDidChange(_ previousTraitCollection: UITraitCollection?) {
        super.traitCollectionDidChange(previousTraitCollection)
        if traitCollection.hasDifferentColorAppearance(comparedTo: previousTraitCollection) {
            applyBackgroundForTraitCollection(traitCollection)
        }
    }

    /// Applies system background (light in light mode, dark in dark mode) to the container and scene.
    func applyBackgroundForColorScheme(_ colorScheme: ColorScheme) {
        let style: UIUserInterfaceStyle = colorScheme == .dark ? .dark : .light
        applyBackgroundForTraitCollection(UITraitCollection(userInterfaceStyle: style))
    }

    private func applyBackgroundForTraitCollection(_ traits: UITraitCollection) {
        let bg = UIColor.systemBackground.resolvedColor(with: traits)
        view.backgroundColor = bg
        scnView?.backgroundColor = bg
        scene.background.contents = bg
    }

    static func resolveScenePath(for sex: Sex) -> String? {
        let candidates: [String]
        switch sex {
        case .male:
            candidates = ["BodyMap/male_body.usdz", "male_body.usdz"]
        case .female:
            candidates = ["BodyMap/female_body.usdz", "female_body.usdz"]
        case .unspecified:
            candidates = ["BodyMap/male_body.usdz", "male_body.usdz", "BodyMap/female_body.usdz", "female_body.usdz"]
        }
        return candidates.first { SCNScene(named: $0) != nil }
    }

    func switchSex(to sex: Sex) {
        guard currentSex != sex else { return }
        currentSex = sex
        loadBody(for: sex)
    }

    private func setupSceneView() {
        scnView = SCNView()
        scnView.translatesAutoresizingMaskIntoConstraints = false
        scnView.antialiasingMode = .multisampling4X
        scnView.preferredFramesPerSecond = 60
        scnView.allowsCameraControl = false
        scnView.showsStatistics = false
        scnView.debugOptions = []
        view.addSubview(scnView)
        NSLayoutConstraint.activate([
            scnView.topAnchor.constraint(equalTo: view.topAnchor),
            scnView.bottomAnchor.constraint(equalTo: view.bottomAnchor),
            scnView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            scnView.trailingAnchor.constraint(equalTo: view.trailingAnchor)
        ])
    }

    private func setupScene() {
        scene = SCNScene()
        scnView.scene = scene
    }

    private func setupCamera() {
        let camera = SCNCamera()
        camera.usesOrthographicProjection = false
        camera.fieldOfView = 32
        camera.zNear = 0.1
        camera.zFar = 200
        cameraNode = SCNNode()
        cameraNode.camera = camera
        cameraNode.name = "MainCamera"
        scene.rootNode.addChildNode(cameraNode)
        scnView.pointOfView = cameraNode
    }

    private func setupLighting() {
        let ambient = SCNNode()
        ambient.light = SCNLight()
        ambient.light?.type = .ambient
        ambient.light?.intensity = 500
        ambient.light?.color = UIColor(white: 0.95, alpha: 1)
        scene.rootNode.addChildNode(ambient)

        let key = SCNNode()
        key.light = SCNLight()
        key.light?.type = .directional
        key.light?.intensity = 900
        key.light?.castsShadow = false
        key.eulerAngles = SCNVector3(-Float.pi / 6, Float.pi / 4, 0)
        scene.rootNode.addChildNode(key)

        let fill = SCNNode()
        fill.light = SCNLight()
        fill.light?.type = .directional
        fill.light?.intensity = 400
        fill.eulerAngles = SCNVector3(-Float.pi / 8, -Float.pi / 4, 0)
        scene.rootNode.addChildNode(fill)

        let rim = SCNNode()
        rim.light = SCNLight()
        rim.light?.type = .directional
        rim.light?.intensity = 300
        rim.light?.color = UIColor(red: 0.85, green: 0.90, blue: 1.0, alpha: 1)
        rim.eulerAngles = SCNVector3(-Float.pi / 6, Float.pi, 0)
        scene.rootNode.addChildNode(rim)
    }

    // MARK: - Load Body

    private func loadBody(for sex: Sex) {
        pivotNode?.removeFromParentNode()
        pivotNode = nil
        meshNode = nil
        markerNode?.removeFromParentNode()
        markerNode = nil

        guard let scenePath = Self.resolveScenePath(for: sex),
              let modelScene = SCNScene(named: scenePath) else { return }

        let nodeName: String
        switch sex {
        case .male: nodeName = "MaleBody"
        case .female: nodeName = "FemaleBody"
        case .unspecified: nodeName = "MaleBody"
        }

        let model: SCNNode
        if let found = modelScene.rootNode.childNode(withName: nodeName, recursively: true) {
            model = found.clone()
        } else {
            let container = SCNNode()
            for child in modelScene.rootNode.childNodes {
                container.addChildNode(child.clone())
            }
            model = container
        }

        setupBodyNode(model)
    }

    private func setupBodyNode(_ node: SCNNode) {
        let skinMaterial = SCNMaterial()
        skinMaterial.diffuse.contents = UIColor(red: 0.71, green: 0.49, blue: 0.32, alpha: 1.0)
        skinMaterial.roughness.contents = NSNumber(value: 0.65)
        skinMaterial.metalness.contents = NSNumber(value: 0.0)
        skinMaterial.lightingModel = .physicallyBased
        skinMaterial.fillMode = .fill
        skinMaterial.isDoubleSided = false

        node.enumerateChildNodes { child, _ in
            child.geometry?.materials = [skinMaterial]
        }
        node.geometry?.materials = [skinMaterial]

        let (minB, maxB) = node.boundingBox
        meshBoundsMin = minB
        meshBoundsMax = maxB

        let pivot = SCNNode()
        pivot.name = "BodyPivot"
        let centerX = (minB.x + maxB.x) / 2.0
        let centerY = (minB.y + maxB.y) / 2.0
        let centerZ = (minB.z + maxB.z) / 2.0
        node.position = SCNVector3(-centerX, -centerY, -centerZ)
        node.name = "BodyMesh"
        pivot.addChildNode(node)
        scene.rootNode.addChildNode(pivot)
        pivotNode = pivot
        meshNode = node

        // Auto-fit: camera distance so full body fits; 2.5x gives a bigger body in frame
        let height = maxB.y - minB.y
        let width = maxB.x - minB.x
        let bodyExtent = max(height, width)
        baseCameraDistance = Float(bodyExtent) * 2.5
        cameraNode.position = SCNVector3(0, 0, baseCameraDistance)
        cameraNode.look(at: SCNVector3(0, 0, 0))

        currentAngleY = 0
        pivotNode.eulerAngles = SCNVector3(0, 0, 0)
    }

    private func setupGestures() {
        let pan = UIPanGestureRecognizer(target: self, action: #selector(handleRotation(_:)))
        pan.minimumNumberOfTouches = 1
        pan.maximumNumberOfTouches = 1
        scnView.addGestureRecognizer(pan)

        let tap = UITapGestureRecognizer(target: self, action: #selector(handleTap(_:)))
        tap.numberOfTapsRequired = 1
        scnView.addGestureRecognizer(tap)
        tap.require(toFail: pan)
    }

    // MARK: - Rotation (360° Y-axis only; no zoom, pan, or tilt)

    @objc private func handleRotation(_ gesture: UIPanGestureRecognizer) {
        guard let pivot = pivotNode else { return }
        let translation = gesture.translation(in: scnView)
        let screenWidth = max(1, scnView.bounds.width)

        switch gesture.state {
        case .began:
            startAngleY = currentAngleY
        case .changed:
            // Only horizontal drag → Y-axis rotation. Vertical drag ignored.
            let yRotation = Float(translation.x / screenWidth) * (2 * .pi)
            currentAngleY = startAngleY + yRotation
            pivot.eulerAngles = SCNVector3(0, currentAngleY, 0)
        default:
            break
        }
    }

    // MARK: - Tap: Mesh hit, marker locked to surface

    @objc private func handleTap(_ gesture: UITapGestureRecognizer) {
        guard let pivotNode else { return }
        let location = gesture.location(in: scnView)

        let hitOptions: [SCNHitTestOption: Any] = [
            .searchMode: SCNHitTestSearchMode.closest.rawValue,
            .boundingBoxOnly: false,
            .ignoreHiddenNodes: true
        ]
        let hitResults = scnView.hitTest(location, options: hitOptions)
        guard let hit = hitResults.first(where: { isValidBodyHit($0) }) else { return }

        let localPos = hit.localCoordinates
        let worldPos = hit.worldCoordinates
        let normal = hit.localNormal
        let faceIndex = hit.faceIndex
        let texCoord = hit.textureCoordinates(withMappingChannel: 0)
        let geoIndex = hit.geometryIndex
        let normPos = normalizePosition(localPos)
        let (partRegion, side) = classifyBodyRegion(localPosition: localPos, normalizedPosition: normPos, surfaceNormal: normal)

        let sidePrefix: String
        switch side {
        case .left: sidePrefix = "Left "
        case .right: sidePrefix = "Right "
        case .center: sidePrefix = ""
        }
        let fullLabel = sidePrefix + partRegion.rawValue

        let result = BodyMeshTapResult(
            worldPosition: worldPos,
            localPosition: localPos,
            surfaceNormal: normal,
            faceIndex: faceIndex,
            textureCoordinates: texCoord,
            geometryIndex: geoIndex,
            normalizedPosition: normPos,
            regionDisplayName: fullLabel,
            mappedBodyRegion: partRegion.toBodyRegion,
            side: side
        )

        placeMarkerOnSurface(hit: hit)

        #if DEBUG
        print("""
        ===================================
        Spot Selected: \(fullLabel)
           World:  (\(String(format: "%.3f", hit.worldCoordinates.x)), \(String(format: "%.3f", hit.worldCoordinates.y)), \(String(format: "%.3f", hit.worldCoordinates.z)))
           Local:  (\(String(format: "%.3f", localPos.x)), \(String(format: "%.3f", localPos.y)), \(String(format: "%.3f", localPos.z)))
           Normal: (\(String(format: "%.3f", normal.x)), \(String(format: "%.3f", normal.y)), \(String(format: "%.3f", normal.z)))
           UV:     (\(String(format: "%.4f", texCoord.x)), \(String(format: "%.4f", texCoord.y)))
           Norm%:  (\(String(format: "%.3f", normPos.x)), \(String(format: "%.3f", normPos.y)), \(String(format: "%.3f", normPos.z)))
           Face:   \(faceIndex)
           Region: \(partRegion.rawValue)  |  Side: \(side.rawValue)
        ===================================
        """)
        #endif

        onSpotSelected?(result)
    }

    private func isValidBodyHit(_ hit: SCNHitTestResult) -> Bool {
        if hit.node.name == "SpotMarker" || hit.node.name == "SpotMarkerRing" { return false }
        var node: SCNNode? = hit.node
        while let n = node {
            if n === pivotNode { return true }
            node = n.parent
        }
        return false
    }

    // MARK: - Surface-locked marker (disc + ring, local coords on hit.node)

    private func placeMarkerOnSurface(hit: SCNHitTestResult) {
        markerNode?.removeFromParentNode()
        markerNode = nil

        let markerRadius: CGFloat = 0.12
        let markerThickness: CGFloat = 0.015
        let disc = SCNCylinder(radius: markerRadius, height: markerThickness)
        let markerMaterial = SCNMaterial()
        markerMaterial.diffuse.contents = UIColor(red: 0.1, green: 0.7, blue: 1.0, alpha: 0.9)
        markerMaterial.emission.contents = UIColor(red: 0.1, green: 0.7, blue: 1.0, alpha: 1.0)
        markerMaterial.emission.intensity = 0.8
        markerMaterial.lightingModel = .physicallyBased
        markerMaterial.isDoubleSided = true
        disc.materials = [markerMaterial]

        let marker = SCNNode(geometry: disc)
        marker.name = "SpotMarker"

        let localPos = hit.localCoordinates
        let localNormal = hit.localNormal
        let normalOffset: Float = 0.02
        marker.position = SCNVector3(
            localPos.x + localNormal.x * normalOffset,
            localPos.y + localNormal.y * normalOffset,
            localPos.z + localNormal.z * normalOffset
        )

        let up = SCNVector3(0, 1, 0)
        let dot = up.x * localNormal.x + up.y * localNormal.y + up.z * localNormal.z
        if abs(dot) < 0.999 {
            let crossX = up.y * localNormal.z - up.z * localNormal.y
            let crossY = up.z * localNormal.x - up.x * localNormal.z
            let crossZ = up.x * localNormal.y - up.y * localNormal.x
            let crossLen = sqrt(crossX * crossX + crossY * crossY + crossZ * crossZ)
            if crossLen > 0.0001 {
                let ax = crossX / crossLen
                let ay = crossY / crossLen
                let az = crossZ / crossLen
                let angle = acos(max(-1, min(1, dot)))
                marker.rotation = SCNVector4(ax, ay, az, angle)
            }
        }

        let pulse = CABasicAnimation(keyPath: "opacity")
        pulse.fromValue = 0.9
        pulse.toValue = 0.5
        pulse.duration = 1.0
        pulse.autoreverses = true
        pulse.repeatCount = .infinity
        pulse.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        marker.addAnimation(pulse, forKey: "pulse")

        let ring = SCNTorus(ringRadius: markerRadius * 1.3, pipeRadius: 0.012)
        let ringMaterial = SCNMaterial()
        ringMaterial.diffuse.contents = UIColor(red: 0.1, green: 0.7, blue: 1.0, alpha: 0.5)
        ringMaterial.emission.contents = UIColor(red: 0.2, green: 0.8, blue: 1.0, alpha: 1.0)
        ringMaterial.emission.intensity = 1.0
        ringMaterial.lightingModel = .physicallyBased
        ringMaterial.isDoubleSided = true
        ring.materials = [ringMaterial]
        let ringNode = SCNNode(geometry: ring)
        ringNode.name = "SpotMarkerRing"
        marker.addChildNode(ringNode)

        hit.node.addChildNode(marker)
        markerNode = marker
    }

    // MARK: - Normalize & 18-region classification

    private func normalizePosition(_ localPos: SCNVector3) -> SIMD3<Float> {
        let rangeX = meshBoundsMax.x - meshBoundsMin.x
        let rangeY = meshBoundsMax.y - meshBoundsMin.y
        let rangeZ = meshBoundsMax.z - meshBoundsMin.z
        return SIMD3<Float>(
            rangeX > 0 ? (localPos.x - meshBoundsMin.x) / rangeX : 0.5,
            rangeY > 0 ? (localPos.y - meshBoundsMin.y) / rangeY : 0.5,
            rangeZ > 0 ? (localPos.z - meshBoundsMin.z) / rangeZ : 0.5
        )
    }

    private func classifyBodyRegion(
        localPosition: SCNVector3,
        normalizedPosition norm: SIMD3<Float>,
        surfaceNormal: SCNVector3
    ) -> (BodyPartRegion, BodySide) {
        let ny = norm.y
        let nx = norm.x
        let nz = norm.z
        let xFromCenter = abs(nx - 0.5)
        let isFront = nz > 0.5

        let side: BodySide
        if xFromCenter < 0.08 {
            side = .center
        } else if nx < 0.5 {
            side = .right
        } else {
            side = .left
        }

        let torsoHalfWidth: Float
        if ny > 0.75 { torsoHalfWidth = 0.18 }
        else if ny > 0.55 { torsoHalfWidth = 0.15 }
        else if ny > 0.40 { torsoHalfWidth = 0.14 }
        else { torsoHalfWidth = 0.10 }
        let isLimb = xFromCenter > torsoHalfWidth

        if ny > 0.92 { return (.scalp, side) }
        if ny > 0.85 { return (.face, side) }
        if ny > 0.80 { return (.neck, side) }

        if ny > 0.55 {
            if isLimb {
                if ny > 0.72 { return (.shoulder, side) }
                if xFromCenter > 0.38 { return (.hand, side) }
                let armProgress = (0.72 - ny) / max(0.01, 0.72 - 0.55)
                if xFromCenter > 0.32 || armProgress > 0.6 { return (.forearm, side) }
                return (.upperArm, side)
            }
            if ny > 0.65 { return isFront ? (.chest, .center) : (.upperBack, .center) }
            return isFront ? (.abdomen, .center) : (.midBack, .center)
        }

        if ny > 0.40 {
            if isLimb {
                if xFromCenter > 0.38 { return (.hand, side) }
                if xFromCenter > 0.28 { return (.forearm, side) }
                return (.upperArm, side)
            }
            if isFront { return (.abdomen, .center) }
            return ny > 0.47 ? (.lowerBack, .center) : (.glutes, side)
        }

        if ny > 0.28 {
            if xFromCenter > 0.35 { return (.hand, side) }
            if ny > 0.37 { return isFront ? (.hip, side) : (.glutes, side) }
            return (.thigh, side)
        }
        if ny > 0.22 { return (.knee, side) }
        if ny > 0.08 { return (.lowerLeg, side) }
        return (.foot, side)
    }

    func clearMarker() {
        markerNode?.removeFromParentNode()
        markerNode = nil
    }

    func resetToFront() {
        currentAngleY = 0
        SCNTransaction.begin()
        SCNTransaction.animationDuration = 0.4
        SCNTransaction.animationTimingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        pivotNode?.eulerAngles = SCNVector3(0, 0, 0)
        SCNTransaction.commit()
    }
}

extension Notification.Name {
    static let resetBodyView = Notification.Name("resetBodyView")
}
