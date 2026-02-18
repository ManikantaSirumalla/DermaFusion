//
//  DFRiskBadge.swift
//  DermFusion
//
//  Badge for displaying classification risk level with color + icon + text.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Risk-level pill used in results and history summaries.
struct DFRiskBadge: View {

    // MARK: - Properties

    let level: RiskLevel
    /// When true, uses smaller font and padding (e.g. for Recents row).
    var compact: Bool = false
    /// When false, only the level text is shown (no icon).
    var showIcon: Bool = true

    // MARK: - Body

    var body: some View {
        HStack(spacing: compact ? 6 : DFDesignSystem.Spacing.xs) {
            if showIcon {
                Image(systemName: level.iconName)
                    .font(compact ? .caption : .body)
            }
            Text(level.displayName)
                .font(compact ? DFDesignSystem.Typography.caption.bold() : DFDesignSystem.Typography.bodyBold)
        }
        .foregroundStyle(DFDesignSystem.Colors.textInverse)
        .padding(.horizontal, compact ? DFDesignSystem.Spacing.sm : DFDesignSystem.Spacing.sm)
        .padding(.vertical, compact ? DFDesignSystem.Spacing.xs : DFDesignSystem.Spacing.xs)
        .background(level.color)
        .clipShape(Capsule())
        .accessibilityElement(children: .combine)
        .accessibilityLabel(level.accessibilityLabel)
    }
}

#Preview("High") {
    DFRiskBadge(level: .high)
}

#Preview("Low") {
    DFRiskBadge(level: .low)
}
