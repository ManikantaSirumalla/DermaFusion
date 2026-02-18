//
//  ScanViewModel.swift
//  DermFusion
//
//  Manages state for the MVP scan flow: review, metadata, analysis, and result.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation
import Combine
import UIKit

@MainActor
final class ScanViewModel: ObservableObject {

    // MARK: - Edge Cases

    // Edge cases handled in this baseline flow:
    // - Analyze blocked until body region is selected.
    // - Minimum loading display duration avoids abrupt transition flashes.
    // - Repeated analyze taps are ignored while request is in progress.

    // MARK: - Published State

    @Published var age: Double = 35
    @Published var sex: Sex = .unspecified
    @Published var selectedRegion: BodyRegion?
    @Published var name: String = ""
    @Published var selectedEthnicity: Ethnicity = .preferNotToSay
    @Published private(set) var isAnalyzing = false
    @Published private(set) var result: DiagnosisResult?
    /// Image after review (and optional crop/tilt). Set when navigating from ImageReviewView.
    @Published var capturedImage: UIImage?

    // MARK: - Public API

    var canAnalyze: Bool {
        selectedRegion != nil && !isAnalyzing
    }

    var metadataInput: MetadataInput? {
        guard let selectedRegion else { return nil }
        return MetadataInput(
            age: Int(age),
            sex: sex,
            bodyRegion: selectedRegion,
            name: name.isEmpty ? nil : name,
            ethnicity: selectedEthnicity == .preferNotToSay ? nil : selectedEthnicity.displayName
        )
    }

    func resetResult() {
        result = nil
    }

    init(capturedImage: UIImage? = nil) {
        self.capturedImage = capturedImage
    }

    func runAnalysis() async {
        guard canAnalyze else { return }
        isAnalyzing = true
        let startTime = Date()
        try? await Task.sleep(nanoseconds: 450_000_000)

        let minimumDuration: TimeInterval = 1.2
        let elapsed = Date().timeIntervalSince(startTime)
        if elapsed < minimumDuration {
            let remaining = minimumDuration - elapsed
            let remainingNanoseconds = UInt64(remaining * 1_000_000_000)
            try? await Task.sleep(nanoseconds: remainingNanoseconds)
        }

        result = .sample
        isAnalyzing = false
    }
}
