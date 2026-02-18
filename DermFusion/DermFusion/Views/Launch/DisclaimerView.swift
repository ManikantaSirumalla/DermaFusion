//
//  DisclaimerView.swift
//  DermFusion
//
//  Full-screen first-launch medical disclaimer gate.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Required first-launch disclaimer that users must acknowledge before using the app.
struct DisclaimerView: View {

    // MARK: - Properties

    let onAccept: () -> Void

    // MARK: - Body

    var body: some View {
        VStack(spacing: DFDesignSystem.Spacing.lg) {
            DFCard {
                VStack(spacing: DFDesignSystem.Spacing.lg) {
                    Image(systemName: DFDesignSystem.Icons.disclaimer)
                        .font(DFDesignSystem.Typography.displayLarge)
                        .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                        .accessibilityHidden(true)

                    Text("Important Disclaimer")
                        .font(DFDesignSystem.Typography.headline)
                        .foregroundStyle(DFDesignSystem.Colors.textPrimary)

                    Text(DFMedicalDisclaimer.fullText)
                        .font(DFDesignSystem.Typography.disclaimer)
                        .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                }
            }

            DFButton(title: "I Understand", variant: .primary, action: onAccept)
                .accessibilityLabel("I understand the medical disclaimer")
        }
        .dfConstrainedContent(maxWidth: 640)
        .padding(DFDesignSystem.Spacing.screenHorizontal)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(DFDesignSystem.Colors.backgroundPrimary)
    }
}

#Preview("Default") {
    DisclaimerView(onAccept: {})
}

#Preview("Dark") {
    DisclaimerView(onAccept: {})
        .preferredColorScheme(.dark)
}
