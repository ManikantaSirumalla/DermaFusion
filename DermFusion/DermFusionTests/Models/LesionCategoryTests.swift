//
//  LesionCategoryTests.swift
//  DermFusionTests
//
//  Unit tests for lesion-category display mapping.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import XCTest
@testable import DermFusion

final class LesionCategoryTests: XCTestCase {
    func test_displayName_forMelanoma_isNonEmpty() {
        XCTAssertFalse(LesionCategory.melanoma.displayName.isEmpty)
    }
}
