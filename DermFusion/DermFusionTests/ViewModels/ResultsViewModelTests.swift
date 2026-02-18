//
//  ResultsViewModelTests.swift
//  DermFusionTests
//
//  Unit tests for results view model output formatting state.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import XCTest
@testable import DermFusion

final class ResultsViewModelTests: XCTestCase {
    func test_init_withSampleResult_setsResult() {
        let viewModel = ResultsViewModel(result: .sample)
        XCTAssertNotNil(viewModel.result)
    }
}
