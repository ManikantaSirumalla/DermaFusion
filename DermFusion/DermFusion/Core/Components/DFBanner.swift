//
//  DFBanner.swift
//  DermFusion
//
//  Reusable banner component for informational, warning, and disclaimer contexts.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

enum DFBannerStyle {
    case info
    case warning
    case disclaimer
}

struct DFBanner: View {
    let text: String
    let style: DFBannerStyle

    var body: some View {
        HStack(alignment: .top, spacing: DFDesignSystem.Spacing.xs) {
            Image(systemName: DFDesignSystem.Icons.disclaimer)
            Text(text)
                .font(DFDesignSystem.Typography.disclaimer)
        }
        .foregroundStyle(DFDesignSystem.Colors.textSecondary)
        .padding(DFDesignSystem.Spacing.sm)
        .background(background)
        .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadiusSmall))
        .overlay(
            RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadiusSmall)
                .stroke(DFDesignSystem.Colors.divider.opacity(0.3), lineWidth: 1)
        )
        .dfShadow(DFDesignSystem.Shadows.card)
    }

    private var background: Color {
        switch style {
        case .info:
            return DFDesignSystem.Colors.backgroundSecondary
        case .warning:
            return DFDesignSystem.Colors.riskModerate.opacity(0.15)
        case .disclaimer:
            return DFDesignSystem.Colors.disclaimerBackground
        }
    }
}
