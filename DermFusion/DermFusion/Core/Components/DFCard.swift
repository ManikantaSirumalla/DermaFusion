//
//  DFCard.swift
//  DermFusion
//
//  Reusable card container that applies DermaFusion surface styling.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// A styled container for grouped content.
struct DFCard<Content: View>: View {

    // MARK: - Properties

    private let content: Content

    // MARK: - Initialization

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    // MARK: - Body

    var body: some View {
        content
            .padding(DFDesignSystem.Spacing.md)
            .background(DFDesignSystem.Colors.backgroundSecondary)
            .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius))
            .overlay(
                RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius)
                    .stroke(DFDesignSystem.Colors.divider.opacity(0.35), lineWidth: 1)
            )
            .dfShadow(DFDesignSystem.Shadows.elevated)
    }
}

#Preview("Default") {
    DFCard {
        Text("DermaFusion card content")
            .font(DFDesignSystem.Typography.bodyRegular)
            .foregroundStyle(DFDesignSystem.Colors.textPrimary)
    }
    .padding(DFDesignSystem.Spacing.md)
}

#Preview("Dark") {
    DFCard {
        Text("DermaFusion card content")
            .font(DFDesignSystem.Typography.bodyRegular)
            .foregroundStyle(DFDesignSystem.Colors.textPrimary)
    }
    .padding(DFDesignSystem.Spacing.md)
    .preferredColorScheme(.dark)
}
