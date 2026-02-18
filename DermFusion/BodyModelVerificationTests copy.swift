import XCTest
import SceneKit

/// Run this test after importing BodyModel into Xcode to verify
/// all 12 region nodes are present and named correctly.
///
/// Add to: DermaFusionTests/BodyModelVerificationTests.swift
final class BodyModelVerificationTests: XCTestCase {

    /// All 12 HAM10000 body regions that must exist as named SCNNodes
    private let requiredRegions = [
        "region_scalp",
        "region_face",
        "region_ear",
        "region_neck",
        "region_chest",
        "region_abdomen",
        "region_back",
        "region_upper_extremity",
        "region_lower_extremity",
        "region_hand",
        "region_foot",
        "region_genital"
    ]

    // MARK: - Scene Loading

    func testBodyModelSceneLoads() throws {
        // Try .scn first, fall back to .dae, then .obj
        let scene = SCNScene(named: "BodyModel.scn")
            ?? SCNScene(named: "BodyModel.dae")
            ?? SCNScene(named: "BodyModel.obj")

        XCTAssertNotNil(scene, """
            BodyModel not found in app bundle.
            Ensure BodyModel.scn (or .dae/.obj) is added to the target.
            """)
    }

    // MARK: - Node Verification

    func testAllTwelveRegionNodesExist() throws {
        let scene = try XCTUnwrap(
            SCNScene(named: "BodyModel.scn")
            ?? SCNScene(named: "BodyModel.dae")
            ?? SCNScene(named: "BodyModel.obj"),
            "BodyModel scene file not found"
        )

        var missingNodes: [String] = []
        var foundNodes: [String] = []

        for regionName in requiredRegions {
            if let node = scene.rootNode.childNode(withName: regionName, recursively: true) {
                foundNodes.append(regionName)

                // Verify node has geometry (is renderable / tappable)
                XCTAssertNotNil(
                    node.geometry,
                    "\(regionName) exists but has no geometry — hit testing won't work"
                )
            } else {
                missingNodes.append(regionName)
            }
        }

        print("── BodyModel Node Verification ──")
        print("Found: \(foundNodes.count)/\(requiredRegions.count)")
        for name in foundNodes {
            let node = scene.rootNode.childNode(withName: name, recursively: true)!
            let vertCount = node.geometry?.sources(for: .vertex).first?.vectorCount ?? 0
            let elemCount = node.geometry?.elements.first?.primitiveCount ?? 0
            print("  ✅ \(name) — \(vertCount) verts, \(elemCount) tris")
        }
        if !missingNodes.isEmpty {
            print("Missing:")
            for name in missingNodes {
                print("  ❌ \(name)")
            }
        }

        XCTAssertEqual(missingNodes.count, 0, """
            Missing \(missingNodes.count) region node(s): \(missingNodes.joined(separator: ", "))
            
            Check that BodyModel was imported correctly and nodes were not
            renamed during .dae → .scn conversion.
            """)
    }

    // MARK: - Geometry Sanity

    func testRegionNodesHaveReasonableGeometry() throws {
        let scene = try XCTUnwrap(
            SCNScene(named: "BodyModel.scn")
            ?? SCNScene(named: "BodyModel.dae"),
            "BodyModel scene file not found"
        )

        for regionName in requiredRegions {
            guard let node = scene.rootNode.childNode(withName: regionName, recursively: true),
                  let geometry = node.geometry else {
                continue  // Caught by testAllTwelveRegionNodesExist
            }

            // Each region should have at least some vertices
            let vertexCount = geometry.sources(for: .vertex).first?.vectorCount ?? 0
            XCTAssertGreaterThan(
                vertexCount, 10,
                "\(regionName) has suspiciously few vertices (\(vertexCount))"
            )

            // Each region should have at least one geometry element (triangle set)
            XCTAssertGreaterThan(
                geometry.elements.count, 0,
                "\(regionName) has no geometry elements"
            )

            // Each region should have a material for highlighting
            XCTAssertGreaterThan(
                geometry.materials.count, 0,
                "\(regionName) has no materials — highlighting won't work"
            )
        }
    }

    // MARK: - Camera & Lights

    func testSceneHasCameraAndLights() throws {
        let scene = try XCTUnwrap(
            SCNScene(named: "BodyModel.scn")
            ?? SCNScene(named: "BodyModel.dae"),
            "BodyModel scene file not found"
        )

        // Camera
        let cameraNode = scene.rootNode.childNode(withName: "CameraNode", recursively: true)
        XCTAssertNotNil(cameraNode?.camera, "Scene should have a camera node named 'CameraNode'")

        // At least one light
        var lightCount = 0
        scene.rootNode.enumerateChildNodes { node, _ in
            if node.light != nil { lightCount += 1 }
        }
        XCTAssertGreaterThan(lightCount, 0, "Scene should have at least one light")
    }

    // MARK: - Hit Test Simulation

    func testHitTestReturnsRegionNames() throws {
        let scene = try XCTUnwrap(
            SCNScene(named: "BodyModel.scn")
            ?? SCNScene(named: "BodyModel.dae"),
            "BodyModel scene file not found"
        )

        // Verify that all region node names match the expected prefix
        var regionNodes: [SCNNode] = []
        scene.rootNode.enumerateChildNodes { node, _ in
            if let name = node.name, name.hasPrefix("region_") {
                regionNodes.append(node)
            }
        }

        XCTAssertEqual(
            regionNodes.count, 12,
            "Expected 12 region_ nodes, found \(regionNodes.count)"
        )

        // Verify each region name maps to a valid BodyRegion key
        let validKeys = Set(requiredRegions.map { $0.replacingOccurrences(of: "region_", with: "") })
        for node in regionNodes {
            let key = node.name!.replacingOccurrences(of: "region_", with: "")
            XCTAssertTrue(
                validKeys.contains(key),
                "Unexpected region node name: \(node.name!) (key: \(key))"
            )
        }
    }
}
