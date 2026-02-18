//
//  DFColors.swift
//  DermFusion
//
//  Semantic color tokens for DermaFusion.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Color tokens used throughout the app.
enum DFColors {

    // MARK: - Brand

    static let brandPrimary = Color("BrandPrimary")
    static let brandSecondary = Color("BrandSecondary")

    // MARK: - Surface

    static let backgroundPrimary = Color(.systemBackground)
    static let backgroundSecondary = Color(.secondarySystemBackground)
    static let surfaceElevated = Color(.tertiarySystemBackground)
    /// Card-style surfaces (e.g. LesionDetailView content card).
    static let cardBackground = Color(.secondarySystemBackground)

    // MARK: - Text

    static let textPrimary = Color.primary
    static let textSecondary = Color.secondary
    static let textTertiary = Color(.tertiaryLabel)
    static let textInverse = Color.white

    // MARK: - Risk

    static let riskLow = Color(.systemGreen)
    static let riskModerate = Color(.systemOrange)
    static let riskHigh = Color(.systemRed)

    // MARK: - Utility

    static let interactive = brandPrimary
    /// Accent for icons and highlights (e.g. detail view insights).
    static let accent = brandPrimary
    static let destructive = Color(.systemRed)
    static let divider = Color(.separator)
    static let disclaimerBackground = Color(.systemYellow).opacity(0.18)
    static let bodyMapHighlight = brandPrimary.opacity(0.35)

    // MARK: - Charts

    static func chartColor(for category: LesionCategory) -> Color {
        switch category {
        case .melanoma:
            return Color(.systemRed)
        case .bcc:
            return Color(.systemOrange)
        case .akiec:
            return Color(.systemYellow)
        case .bkl:
            return Color(.systemTeal)
        case .nevus:
            return Color(.systemBlue)
        case .dermatofibroma:
            return Color(.systemIndigo)
        case .vascular:
            return Color(.systemPurple)
        }
    }
}
