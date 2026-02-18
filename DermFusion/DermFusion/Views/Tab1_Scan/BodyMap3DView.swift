//
//  BodyMap3DView.swift
//  DermFusion
//
//  SceneKit-powered 3D body map for region selection.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SceneKit
import SwiftUI
import UIKit

/// Wraps `SCNView` for SwiftUI and returns region selections through bindings.
struct BodyMap3DView: UIViewRepresentable {

    // MARK: - Properties

    let sex: Sex
    @Binding var selectedRegion: BodyRegion?

    // MARK: - UIViewRepresentable

    func makeCoordinator() -> Coordinator {
        Coordinator(selectedRegion: $selectedRegion)
    }

    func makeUIView(context: Context) -> SCNView {
        let sceneView = SCNView()
        sceneView.backgroundColor = .clear
        sceneView.allowsCameraControl = true
        sceneView.defaultCameraController.interactionMode = .orbitTurntable
        sceneView.defaultCameraController.inertiaEnabled = true
        sceneView.antialiasingMode = .multisampling4X
        sceneView.isJitteringEnabled = true
        sceneView.preferredFramesPerSecond = 60

        if let scene = Self.loadBodyScene(for: sex) {
            sceneView.scene = scene
            Self.applyScenePolish(sceneView: sceneView, scene: scene)
            context.coordinator.configureRegionNodeMapping(in: scene.rootNode)
            context.coordinator.cacheInitialMaterials(in: scene.rootNode)
        } else {
            sceneView.scene = SCNScene()
        }

        let tapGesture = UITapGestureRecognizer(
            target: context.coordinator,
            action: #selector(Coordinator.handleTap(_:))
        )
        sceneView.addGestureRecognizer(tapGesture)
        context.coordinator.sceneView = sceneView
        context.coordinator.captureInitialCameraState(from: sceneView.pointOfView)
        sceneView.delegate = context.coordinator
        return sceneView
    }

    func updateUIView(_ uiView: SCNView, context: Context) {
        context.coordinator.updateSelectionVisuals(for: selectedRegion)
    }

    /// Loads the first available SceneKit-compatible body model format.
    static func loadBodyScene(for sex: Sex) -> SCNScene? {
        guard let descriptor = resolvedSceneDescriptor(for: sex) else { return nil }
        return SCNScene(named: descriptor.scenePath)
    }

    private static func resolvedSceneDescriptor(for sex: Sex) -> (scenePath: String, displayName: String)? {
        let candidates: [(scenePath: String, displayName: String)]
        switch sex {
        case .male:
            candidates = modelCandidates(baseNames: ["BodyMap/male_body", "male_body"], displayName: "male_body")
        case .female:
            candidates = modelCandidates(baseNames: ["BodyMap/female_body", "female_body"], displayName: "female_body")
        case .unspecified:
            candidates = modelCandidates(
                baseNames: ["BodyMap/male_body", "male_body", "BodyMap/female_body", "female_body"],
                displayName: "default gender mesh"
            ) + legacyBodyModelCandidates()
        }

        for candidate in candidates where SCNScene(named: candidate.scenePath) != nil {
            return candidate
        }
        return nil
    }

    private static func modelCandidates(
        baseNames: [String],
        displayName: String
    ) -> [(scenePath: String, displayName: String)] {
        var candidates: [(scenePath: String, displayName: String)] = []
        for baseName in baseNames {
            for ext in ["usdz", "scn", "dae", "obj"] {
                candidates.append(("\(baseName).\(ext)", "\(displayName).\(ext)"))
            }
        }
        return candidates
    }

    private static func legacyBodyModelCandidates() -> [(scenePath: String, displayName: String)] {
        [
            ("BodyMap/BodyModel.scn", "legacy BodyModel.scn"),
            ("BodyMap/BodyModel.dae", "legacy BodyModel.dae"),
            ("BodyMap/BodyModel.obj", "legacy BodyModel.obj"),
            ("BodyModel.scn", "legacy BodyModel.scn"),
            ("BodyModel.dae", "legacy BodyModel.dae"),
            ("BodyModel.obj", "legacy BodyModel.obj")
        ]
    }

    private static func applyScenePolish(sceneView: SCNView, scene: SCNScene) {
        sceneView.autoenablesDefaultLighting = false
        scene.lightingEnvironment.intensity = 1.0

        let rootNode = scene.rootNode
        applyHumanMannequinAppearance(rootNode: rootNode)
        let hasCamera = rootNode.childNode(withName: "CameraNode", recursively: true)?.camera != nil
        if !hasCamera {
            let cameraNode = SCNNode()
            cameraNode.name = "CameraNode"
            cameraNode.camera = SCNCamera()
            cameraNode.camera?.fieldOfView = 36
            cameraNode.position = SCNVector3(0, 1.0, 2.5)
            rootNode.addChildNode(cameraNode)
            sceneView.pointOfView = cameraNode
        }

        var lightCount = 0
        rootNode.enumerateChildNodes { node, _ in
            if node.light != nil {
                lightCount += 1
            }
        }

        if lightCount == 0 {
            let ambientNode = SCNNode()
            ambientNode.light = SCNLight()
            ambientNode.light?.type = .ambient
            ambientNode.light?.intensity = 650
            ambientNode.light?.color = UIColor(white: 0.95, alpha: 1.0)
            rootNode.addChildNode(ambientNode)

            let keyLight = SCNNode()
            keyLight.light = SCNLight()
            keyLight.light?.type = .directional
            keyLight.light?.intensity = 950
            keyLight.eulerAngles = SCNVector3(-0.9, 0.5, 0)
            rootNode.addChildNode(keyLight)
        }

        configureCameraInteraction(sceneView: sceneView)
    }

    /// Allows horizontal rotation only and locks zoom distance.
    private static func configureCameraInteraction(sceneView: SCNView) {
        sceneView.allowsCameraControl = true
        sceneView.defaultCameraController.interactionMode = .orbitTurntable
        sceneView.defaultCameraController.inertiaEnabled = true
        sceneView.defaultCameraController.minimumVerticalAngle = 0
        sceneView.defaultCameraController.maximumVerticalAngle = 0
    }

    private static func cameraDistance(from pointOfView: SCNNode?) -> Float {
        guard let pointOfView else { return 2.5 }
        let x = pointOfView.position.x
        let y = pointOfView.position.y
        let z = pointOfView.position.z
        return max(0.1, sqrt((x * x) + (y * y) + (z * z)))
    }

    /// Gives the body model a clinically neutral, human-like mannequin finish.
    private static func applyHumanMannequinAppearance(rootNode: SCNNode) {
        let neutralSkinTone = UIColor(red: 0.86, green: 0.78, blue: 0.72, alpha: 1.0)
        rootNode.enumerateChildNodes { node, _ in
            guard let materials = node.geometry?.materials else { return }
            for material in materials {
                material.lightingModel = .physicallyBased
                material.diffuse.contents = neutralSkinTone
                material.metalness.contents = 0.0
                material.roughness.contents = 0.9
                material.specular.contents = UIColor(white: 0.12, alpha: 1.0)
                material.emission.contents = UIColor.clear
                material.shininess = 0.02
                material.isDoubleSided = true
            }
        }
    }
}

// MARK: - Coordinator

extension BodyMap3DView {
    final class Coordinator: NSObject, SCNSceneRendererDelegate {
        @Binding private var selectedRegion: BodyRegion?
        weak var sceneView: SCNView?
        private var lockedCameraDistance: Float?
        private var lockedCameraHeight: Float?
        private var lockedInitialAzimuth: Float?
        private let maxAzimuthDeltaRadians: Float = .pi / 3.0 // +/-60 degrees, not full 360

        private var defaultDiffuseByNode: [ObjectIdentifier: [Any]] = [:]
        private var regionNodes: [BodyRegion: [SCNNode]] = [:]

        init(selectedRegion: Binding<BodyRegion?>) {
            _selectedRegion = selectedRegion
        }

        func captureInitialCameraState(from pointOfView: SCNNode?) {
            guard let pointOfView else { return }
            let position = pointOfView.position
            lockedCameraDistance = max(
                0.1,
                sqrt((position.x * position.x) + (position.y * position.y) + (position.z * position.z))
            )
            lockedCameraHeight = position.y
            lockedInitialAzimuth = atan2(position.x, position.z)
        }

        func configureRegionNodeMapping(in rootNode: SCNNode) {
            regionNodes = [:]
            rootNode.enumerateChildNodes { node, _ in
                guard node.geometry != nil else { return }
                guard let region = regionForNodeIdentifiers(node) else { return }
                regionNodes[region, default: []].append(node)
            }
        }

        func cacheInitialMaterials(in rootNode: SCNNode) {
            rootNode.enumerateChildNodes { node, _ in
                guard let materials = node.geometry?.materials, !materials.isEmpty else {
                    return
                }
                let identifier = ObjectIdentifier(node)
                defaultDiffuseByNode[identifier] = materials.map { $0.diffuse.contents as Any }
            }
            if regionNodes.isEmpty {
                for region in BodyRegion.allCases {
                    guard let node = rootNode.childNode(withName: region.sceneNodeName, recursively: true) else {
                        continue
                    }
                    regionNodes[region, default: []].append(node)
                }
            }
        }

        @objc func handleTap(_ gesture: UITapGestureRecognizer) {
            guard let sceneView else { return }
            let location = gesture.location(in: sceneView)
            let hitResults = sceneView.hitTest(location, options: [:])
            for result in hitResults {
                if let region = regionForTappedNode(result.node) {
                    selectedRegion = region
                    updateSelectionVisuals(for: region)
                    return
                }
            }
        }

        private func regionForTappedNode(_ node: SCNNode) -> BodyRegion? {
            var current: SCNNode? = node
            while let activeNode = current {
                if let region = regionForNodeIdentifiers(activeNode) {
                    regionNodes[region, default: []].append(activeNode)
                    return region
                }
                current = activeNode.parent
            }
            return nil
        }

        private func regionForNodeIdentifiers(_ node: SCNNode) -> BodyRegion? {
            if let nodeName = node.name, let region = BodyRegion(sceneNodeName: nodeName) {
                return region
            }
            if let geometryName = node.geometry?.name, let region = BodyRegion(sceneNodeName: geometryName) {
                return region
            }
            if let materialName = node.geometry?.firstMaterial?.name, let region = BodyRegion(sceneNodeName: materialName) {
                return region
            }
            return nil
        }

        func updateSelectionVisuals(for region: BodyRegion?) {
            for (bodyRegion, nodes) in regionNodes {
                for node in nodes {
                    guard let materials = node.geometry?.materials else { continue }
                    let identifier = ObjectIdentifier(node)
                    for (index, material) in materials.enumerated() {
                        if bodyRegion == region {
                            material.diffuse.contents = UIColor.systemBlue.withAlphaComponent(0.45)
                            material.emission.contents = UIColor.systemBlue.withAlphaComponent(0.18)
                        } else {
                            let originals = defaultDiffuseByNode[identifier]
                            material.diffuse.contents = originals?[safe: index] ?? UIColor.systemGray5
                            material.emission.contents = UIColor.clear
                        }
                    }
                }
            }
        }

        func renderer(_ renderer: any SCNSceneRenderer, updateAtTime time: TimeInterval) {
            guard let sceneView, renderer === sceneView else { return }
            guard let pointOfView = sceneView.pointOfView else { return }
            guard
                let lockedCameraDistance,
                let lockedCameraHeight,
                let lockedInitialAzimuth
            else {
                return
            }

            let current = pointOfView.position
            let currentAzimuth = atan2(current.x, current.z)
            let minAzimuth = lockedInitialAzimuth - maxAzimuthDeltaRadians
            let maxAzimuth = lockedInitialAzimuth + maxAzimuthDeltaRadians
            let clampedAzimuth = min(max(currentAzimuth, minAzimuth), maxAzimuth)

            let horizontalRadiusSquared = max(
                0.001,
                (lockedCameraDistance * lockedCameraDistance) - (lockedCameraHeight * lockedCameraHeight)
            )
            let horizontalRadius = sqrt(horizontalRadiusSquared)
            let targetX = sin(clampedAzimuth) * horizontalRadius
            let targetZ = cos(clampedAzimuth) * horizontalRadius

            let currentDistance = max(
                0.001,
                sqrt((current.x * current.x) + (current.y * current.y) + (current.z * current.z))
            )
            let needsUpdate =
                abs(currentDistance - lockedCameraDistance) > 0.0005 ||
                abs(current.y - lockedCameraHeight) > 0.0005 ||
                abs(current.x - targetX) > 0.0005 ||
                abs(current.z - targetZ) > 0.0005

            guard needsUpdate else { return }
            pointOfView.position = SCNVector3(targetX, lockedCameraHeight, targetZ)
        }
    }
}

