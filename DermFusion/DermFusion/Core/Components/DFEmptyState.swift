//
//  DFEmptyState.swift
//  DermFusion
//
//  Empty-state component with icon, explanatory text, and optional action.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

struct DFEmptyState: View {
    let icon: String
    let title: String
    let message: String
    let actionTitle: String?
    let action: (() -> Void)?

    var body: some View {
        DFCard {
            VStack(spacing: DFDesignSystem.Spacing.md) {
                Image(systemName: icon)
                    .font(DFDesignSystem.Typography.displayLarge)
                    .foregroundStyle(DFDesignSystem.Colors.brandSecondary)
                Text(title)
                    .font(DFDesignSystem.Typography.headline)
                Text(message)
                    .font(DFDesignSystem.Typography.bodyRegular)
                    .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                    .multilineTextAlignment(.center)
                if let actionTitle, let action {
                    DFButton(title: actionTitle, variant: .primary, action: action)
                }
            }
        }
    }
}
