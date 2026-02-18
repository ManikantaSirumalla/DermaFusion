//
//  ScanRecord.swift
//  DermFusion
//
//  Persistent scan record data model used by history and export flows.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation
import SwiftData

/// Local scan record persisted on device.
@Model
final class ScanRecord {
    var id: UUID
    var timestamp: Date
    var imageData: Data
    var age: Int?
    var sex: String
    var lesionLocation: String
    var predictions: [String: Double]
    var primaryDiagnosis: String
    var riskLevel: String
    var gradcamImageData: Data?
    var userNotes: String?

    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        imageData: Data,
        age: Int?,
        sex: String,
        lesionLocation: String,
        predictions: [String: Double],
        primaryDiagnosis: String,
        riskLevel: String,
        gradcamImageData: Data? = nil,
        userNotes: String? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.imageData = imageData
        self.age = age
        self.sex = sex
        self.lesionLocation = lesionLocation
        self.predictions = predictions
        self.primaryDiagnosis = primaryDiagnosis
        self.riskLevel = riskLevel
        self.gradcamImageData = gradcamImageData
        self.userNotes = userNotes
    }
}

// MARK: - LesionCategory helpers

extension ScanRecord {

    /// Probability for the given lesion category (looks up by rawValue then displayName).
    func probability(for category: LesionCategory) -> Double {
        predictions[category.rawValue]
            ?? predictions[category.displayName]
            ?? 0
    }

    /// Primary diagnosis as LesionCategory for UI binding (e.g. probability gauge selection).
    var primaryDiagnosisLesionCategory: LesionCategory? {
        LesionCategory.allCases.first { $0.displayName == primaryDiagnosis }
            ?? LesionCategory(rawValue: primaryDiagnosis.lowercased())
    }
}
