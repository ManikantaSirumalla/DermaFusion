//
//  ResultsViewModel.swift
//  DermFusion
//
//  View model that formats result data and actions for the results screen.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Combine
import Foundation

@MainActor
final class ResultsViewModel: ObservableObject {

    // MARK: - Properties

    @Published private(set) var result: DiagnosisResult?
    @Published var showGradCAM = false

    // MARK: - Initialization

    init(result: DiagnosisResult? = nil) {
        self.result = result
    }
}
