//
//  ExpandableGradCAMHeroView.swift
//  DermFusion
//
//  Expandable hero image container for result and scan-detail screens.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// A tappable hero image block that expands to a full-screen viewer.
struct ExpandableGradCAMHeroView: View {
    @Binding var showGradCAM: Bool
    let title: String

    @State private var isExpanded = false

    var body: some View {
        Button {
            isExpanded = true
        } label: {
            GradCAMOverlayView(showGradCAM: $showGradCAM)
                .frame(maxWidth: .infinity)
                .frame(height: 300)
                .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius))
        }
        .buttonStyle(.plain)
        .fullScreenCover(isPresented: $isExpanded) {
            ExpandedGradCAMViewer(showGradCAM: $showGradCAM, title: title)
        }
    }
}

private struct ExpandedGradCAMViewer: View {
    @Binding var showGradCAM: Bool
    let title: String

    @Environment(\.dismiss) private var dismiss
    @State private var zoomScale: CGFloat = 1

    var body: some View {
        ZStack(alignment: .topTrailing) {
            Color.black.ignoresSafeArea()

            VStack(spacing: DFDesignSystem.Spacing.md) {
                GradCAMOverlayView(showGradCAM: $showGradCAM)
                    .scaleEffect(zoomScale)
                    .gesture(
                        MagnificationGesture()
                            .onChanged { value in
                                zoomScale = min(max(value, 1), 4)
                            }
                    )
                    .padding(DFDesignSystem.Spacing.md)

                Toggle("Show GradCAM", isOn: $showGradCAM)
                    .tint(DFDesignSystem.Colors.brandPrimary)
                    .foregroundStyle(DFDesignSystem.Colors.textInverse)
                    .padding(.horizontal, DFDesignSystem.Spacing.lg)
                    .accessibilityLabel("Toggle expanded GradCAM overlay")
            }

            Button {
                dismiss()
            } label: {
                Image(systemName: "xmark.circle.fill")
                    .font(DFDesignSystem.Typography.displayMedium)
                    .foregroundStyle(DFDesignSystem.Colors.textInverse.opacity(0.9))
                    .padding(DFDesignSystem.Spacing.md)
            }
        }
        .accessibilityLabel("Expanded image for \(title)")
    }
}

