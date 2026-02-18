//
//  DFMedicalDisclaimer.swift
//  DermFusion
//
//  Centralized medical disclaimer text and preference keys.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

/// Medical-context strings that must remain consistent across the app.
enum DFMedicalDisclaimer {

    // MARK: - Keys

    static let acceptanceKey = "hasAcceptedDisclaimer"

    // MARK: - Text

    static let fullText = "DermaFusion is a research and educational tool. It is NOT a medical device and has NOT been approved by the FDA or any regulatory authority. Results should NOT be used for self-diagnosis. Always consult a qualified dermatologist for skin concerns."

    static let bannerText = "Research tool only. Classification results suggest patterns and are not medical advice."
}
