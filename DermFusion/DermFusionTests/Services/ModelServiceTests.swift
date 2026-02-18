//
//  ModelServiceTests.swift
//  DermFusionTests
//
//  Model inference parity test scaffold.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import XCTest
@testable import DermFusion

final class ModelServiceTests: XCTestCase {
    func test_classify_placeholder() async throws {
        let service = ModelService()
        let result = try await service.classify()
        XCTAssertEqual(result.primaryDiagnosis, .nevus)
    }
}
