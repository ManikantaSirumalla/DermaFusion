//
//  DFButton.swift
//  DermFusion
//
//  Reusable button styles aligned to DermaFusion design tokens.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Supported DermaFusion button variants.
enum DFButtonVariant {
    case primary
    case secondary
    case destructive
}

/// Reusable button for consistent app actions.
struct DFButton: View {

    // MARK: - Properties

    let title: String
    let variant: DFButtonVariant
    let action: () -> Void

    // MARK: - Body

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(DFDesignSystem.Typography.bodyBold)
                .frame(maxWidth: .infinity, minHeight: DFDesignSystem.Spacing.touchTarget)
        }
        .buttonStyle(.plain)
        .foregroundStyle(foregroundColor)
        .padding(.horizontal, DFDesignSystem.Spacing.md)
        .padding(.vertical, DFDesignSystem.Spacing.sm)
        .background(backgroundColor)
        .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius))
    }

    // MARK: - Private Helpers

    private var foregroundColor: Color {
        switch variant {
        case .primary, .destructive:
            return DFDesignSystem.Colors.textInverse
        case .secondary:
            return DFDesignSystem.Colors.brandPrimary
        }
    }

    private var backgroundColor: Color {
        switch variant {
        case .primary:
            return DFDesignSystem.Colors.brandPrimary
        case .secondary:
            return DFDesignSystem.Colors.backgroundSecondary
        case .destructive:
            return DFDesignSystem.Colors.riskHigh
        }
    }
}

#Preview("Primary") {
    DFButton(title: "Analyze", variant: .primary, action: {})
        .padding(DFDesignSystem.Spacing.md)
}

#Preview("Secondary") {
    DFButton(title: "Choose Photo", variant: .secondary, action: {})
        .padding(DFDesignSystem.Spacing.md)
}
