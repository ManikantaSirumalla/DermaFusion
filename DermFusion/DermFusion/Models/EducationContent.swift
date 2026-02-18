//
//  EducationContent.swift
//  DermFusion
//
//  Codable models for bundled educational content and attributions.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

/// Root bundled educational document loaded from `EducationalContent.json`.
struct EducationContentDocument: Decodable, Sendable {
    let version: String
    let lastUpdated: String
    let sources: [EducationSource]
    let abcde: [ABCDEPoint]
    let dermoscopyBasics: [EducationParagraph]
    let lesions: [LesionEducation]
}

/// Metadata describing an educational source used in bundled content.
struct EducationSource: Decodable, Sendable, Identifiable {
    let id: String
    let title: String
    let url: String
    let license: String
}

/// A single point in the ABCDE melanoma awareness rule.
struct ABCDEPoint: Decodable, Sendable, Identifiable {
    let letter: String
    let title: String
    let description: String

    var id: String { letter }
}

/// A paragraph block for educational text sections.
struct EducationParagraph: Decodable, Sendable, Identifiable {
    let id: String
    let title: String
    let body: String
}

/// Educational content for one lesion class.
struct LesionEducation: Decodable, Sendable, Identifiable {
    let categoryCode: String
    let title: String
    let severity: String
    let overview: String
    let prevalenceNote: String
    let keyVisualFeatures: [String]
    let dermoscopicClues: [String]
    let demographics: DemographicsNote
    let examples: [EducationalImageReference]
    let references: [EducationReference]

    var id: String { categoryCode }

    var category: LesionCategory? {
        LesionCategory(rawValue: categoryCode)
    }
}

/// Demographic tendencies and common body site notes.
struct DemographicsNote: Decodable, Sendable {
    let ageRange: String
    let sexTendency: String
    let commonSites: [String]
}

/// Example educational image metadata bundled with the app.
struct EducationalImageReference: Decodable, Sendable, Identifiable {
    let id: String
    let imageName: String
    let caption: String
    let license: String
    let sourceURL: String
    let attribution: String
}

/// Citation item supporting bundled educational claims.
struct EducationReference: Decodable, Sendable, Identifiable {
    let id: String
    let label: String
    let url: String
}

