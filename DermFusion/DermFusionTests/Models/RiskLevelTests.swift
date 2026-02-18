//
//  RiskLevelTests.swift
//  DermFusionTests
//
//  Unit tests for risk-level display and semantics.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import XCTest
@testable import DermFusion

final class RiskLevelTests: XCTestCase {
    func test_highRisk_accessibilityLabel_containsConsult() {
        XCTAssertTrue(RiskLevel.high.accessibilityLabel.contains("consult"))
    }
}
