//
//  RiskLevel.swift
//  DermFusion
//
//  Risk-level representation and display/accessibility behavior.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Classification risk bucket shown to users.
enum RiskLevel: String, CaseIterable, Sendable {
    case low
    case moderate
    case high

    var displayName: String {
        switch self {
        case .low:
            return "Low Risk"
        case .moderate:
            return "Moderate Risk"
        case .high:
            return "High Risk"
        }
    }

    var color: Color {
        switch self {
        case .low:
            return DFDesignSystem.Colors.riskLow
        case .moderate:
            return DFDesignSystem.Colors.riskModerate
        case .high:
            return DFDesignSystem.Colors.riskHigh
        }
    }

    var iconName: String {
        switch self {
        case .low:
            return DFDesignSystem.Icons.riskLow
        case .moderate:
            return DFDesignSystem.Icons.riskModerate
        case .high:
            return DFDesignSystem.Icons.riskHigh
        }
    }

    var accessibilityLabel: String {
        switch self {
        case .low:
            return "Low risk classification"
        case .moderate:
            return "Moderate risk classification, consult a dermatologist for confirmation"
        case .high:
            return "High risk classification, consult a dermatologist"
        }
    }
}
