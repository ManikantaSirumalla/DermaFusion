//
//  ClassificationResult.swift
//  DermFusion
//
//  View model for LesionDetailView: classification output plus optional image,
//  metadata, and derived learnings. Build from DiagnosisResult or ScanRecord.
//

import Foundation
import UIKit

/// Input to LesionDetailView: diagnosis, confidence, image, date, location, predictions, learnings.
struct ClassificationResult {
    let primaryDiagnosis: LesionCategory
    let confidence: Double
    let capturedImage: UIImage?
    let date: Date
    let anatomicalLocation: AnatomicalLocation?
    let allPredictions: [Prediction]
    let learnings: [LearningItem]

    /// Build from live scan flow (DiagnosisResult + optional image/metadata).
    init(
        result: DiagnosisResult,
        capturedImage: UIImage? = nil,
        metadata: MetadataInput? = nil,
        date: Date = Date()
    ) {
        primaryDiagnosis = result.primaryDiagnosis
        confidence = result.topProbability
        self.capturedImage = capturedImage
        self.date = date
        anatomicalLocation = metadata.map { AnatomicalLocation(bodyRegion: $0.bodyRegion) }
        allPredictions = result.probabilities
            .sorted { $0.value > $1.value }
            .map { Prediction(label: $0.key.displayName, confidence: $0.value) }
        learnings = Self.learningsFrom(result: result)
    }

    /// Build from a saved ScanRecord (e.g. History → detail).
    init?(record: ScanRecord) {
        guard let category = LesionCategory.allCases.first(where: { $0.displayName == record.primaryDiagnosis })
            ?? LesionCategory.allCases.first(where: { $0.rawValue == record.primaryDiagnosis.lowercased() }) else {
            return nil
        }
        let topProb = record.predictions.values.max() ?? 0
        primaryDiagnosis = category
        confidence = topProb
        capturedImage = record.imageData.isEmpty ? nil : UIImage(data: record.imageData)
        date = record.timestamp
        anatomicalLocation = BodyRegion.allCases.first(where: { $0.displayName == record.lesionLocation })
            .map { AnatomicalLocation(bodyRegion: $0) }
        let fromRecord = record.predictions
            .sorted { $0.value > $1.value }
            .compactMap { key, value in
                LesionCategory(rawValue: key).map { Prediction(label: $0.displayName, confidence: value) }
            }
        allPredictions = fromRecord.isEmpty
            ? [Prediction(label: record.primaryDiagnosis, confidence: topProb)]
            : fromRecord
        learnings = Self.learningsFrom(riskLevel: record.riskLevel, primary: record.primaryDiagnosis)
    }

    private static func learningsFrom(result: DiagnosisResult) -> [LearningItem] {
        learningsFrom(riskLevel: result.riskLevel.rawValue, primary: result.primaryDiagnosis.displayName)
    }

    private static func learningsFrom(riskLevel: String, primary: String) -> [LearningItem] {
        var items: [LearningItem] = []
        let low = riskLevel.lowercased()
        if low == "low" || low == "low risk" {
            items.append(LearningItem(
                icon: "checkmark.shield.fill",
                title: "Low risk assessment",
                description: "Model suggests benign features. Still recommend clinical correlation."
            ))
        } else if low == "high" || low == "high risk" {
            items.append(LearningItem(
                icon: "exclamationmark.triangle.fill",
                title: "Higher risk flag",
                description: "Consider prompt dermatology evaluation for \(primary)."
            ))
        }
        items.append(LearningItem(
            icon: "book.closed.fill",
            title: "Educational only",
            description: "This result is not a diagnosis. See Learn tab for condition details."
        ))
        return items
    }
}

/// Single class prediction for the breakdown list.
struct Prediction {
    let label: String
    let confidence: Double
}

/// Clinical insight card (icon + title + description).
struct LearningItem {
    let icon: String
    let title: String
    let description: String
}

/// Anatomical location for the detail view (from BodyRegion).
struct AnatomicalLocation {
    let displayName: String
    let regionDescription: String

    init(bodyRegion: BodyRegion) {
        displayName = bodyRegion.displayName
        regionDescription = "Lesion location recorded as \(bodyRegion.displayName)."
    }
}
