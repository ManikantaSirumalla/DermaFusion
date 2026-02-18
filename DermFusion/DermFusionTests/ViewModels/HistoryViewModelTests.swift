//
//  HistoryViewModelTests.swift
//  DermFusionTests
//
//  Unit tests for history view model empty state behavior.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import XCTest
@testable import DermFusion

final class HistoryViewModelTests: XCTestCase {
    func test_initialState_isEmpty() {
        let viewModel = HistoryViewModel()
        XCTAssertTrue(viewModel.isEmpty)
    }
}
