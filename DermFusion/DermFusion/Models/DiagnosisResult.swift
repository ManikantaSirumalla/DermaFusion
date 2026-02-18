//
//  DiagnosisResult.swift
//  DermFusion
//
//  Lightweight result model used by the flow-aligned UI placeholders.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

/// Classification output used to render the results screen.
struct DiagnosisResult: Sendable {
    let probabilities: [LesionCategory: Double]
    let primaryDiagnosis: LesionCategory
    let topProbability: Double
    let riskLevel: RiskLevel

    static let sample = DiagnosisResult(
        probabilities: [
            .nevus: 0.873,
            .melanoma: 0.042,
            .bcc: 0.028,
            .bkl: 0.025,
            .akiec: 0.017,
            .dermatofibroma: 0.009,
            .vascular: 0.006,
        ],
        primaryDiagnosis: .nevus,
        topProbability: 0.873,
        riskLevel: .low
    )
}
