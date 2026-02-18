//
//  ScanViewModelTests.swift
//  DermFusionTests
//
//  Unit tests for scan-flow view model state transitions.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import XCTest
@testable import DermFusion

final class ScanViewModelTests: XCTestCase {
    @MainActor
    func test_analyze_withoutLocation_doesNotProduceResult() async {
        let viewModel = ScanViewModel()
        await viewModel.runAnalysis()
        let result = viewModel.result
        XCTAssertNil(result)
    }
}
