//
//  ProbabilityCardRow.swift
//  DermFusion
//
//  Probability row: circular % badge, class name, Select/Selected button.
//  Supports card style (rounded, shadow) and list style (plain rows, thin dividers).
//

import SwiftUI

/// Ring color by probability band (high = green, medium = yellow/orange, low = red).
private func ringColor(for probability: Double) -> Color {
    switch probability {
    case 0.5...: return DFDesignSystem.Colors.riskLow
    case 0.2..<0.5: return DFDesignSystem.Colors.riskModerate
    default: return DFDesignSystem.Colors.riskHigh
    }
}

// MARK: - List-style button views (no row border; only button changes)

/// Outlined button for unselected state: border + transparent background.
private struct OutlinedSelectButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text("Select")
                .font(DFDesignSystem.Typography.caption)
                .fontWeight(.medium)
                .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                .padding(.horizontal, DFDesignSystem.Spacing.sm)
                .padding(.vertical, DFDesignSystem.Spacing.xs)
                .background(Color.clear)
                .overlay(
                    Capsule()
                        .stroke(DFDesignSystem.Colors.brandPrimary, lineWidth: 1)
                )
        }
        .buttonStyle(.plain)
    }
}

/// Filled button for selected state: solid background + checkmark.
private struct FilledSelectedButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 4) {
                Image(systemName: "checkmark.circle.fill")
                    .font(.caption)
                Text("Selected")
                    .font(DFDesignSystem.Typography.caption)
                    .fontWeight(.medium)
            }
            .foregroundStyle(DFDesignSystem.Colors.textInverse)
            .padding(.horizontal, DFDesignSystem.Spacing.sm)
            .padding(.vertical, DFDesignSystem.Spacing.xs)
            .background(DFDesignSystem.Colors.brandPrimary)
            .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

/// One row in the probabilities list: circular % badge, display name, optional subtitle, Select/Selected button.
struct ProbabilityCardRow: View {

    let displayName: String
    let subtitle: String
    let probability: Double
    let isSelected: Bool
    let onSelect: () -> Void
    /// When false, Select button is hidden and row is read-only.
    var selectable: Bool = true
    /// When true, uses smaller font, smaller badge, and smaller button; button shows "Selected" when isSelected.
    var compact: Bool = false
    /// When true, no card background/stroke/shadow; use with thin dividers in container. Button uses Outlined/Filled styles.
    var listStyle: Bool = false

    private var badgeSize: CGFloat {
        if listStyle { return 40 }
        return compact ? 40 : 52
    }
    private var ringLineWidth: CGFloat { compact || listStyle ? 4 : 5 }

    var body: some View {
        let rowContent = HStack(alignment: .center, spacing: listStyle ? DFDesignSystem.Spacing.sm : (compact ? DFDesignSystem.Spacing.sm : DFDesignSystem.Spacing.md)) {
            circularBadge
            VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.xxs) {
                Text(displayName)
                    .font(listStyle ? DFDesignSystem.Typography.caption : (compact ? DFDesignSystem.Typography.caption : DFDesignSystem.Typography.bodyBold))
                    .fontWeight(listStyle ? .medium : (compact ? .medium : .bold))
                    .foregroundStyle(DFDesignSystem.Colors.textPrimary)
                if !subtitle.isEmpty {
                    Text(subtitle)
                        .font(DFDesignSystem.Typography.caption)
                        .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            if selectable {
                selectButton
            }
        }
        .padding(.vertical, listStyle ? 12 : (compact ? DFDesignSystem.Spacing.sm : DFDesignSystem.Spacing.md))
        .padding(.horizontal, listStyle ? 0 : (compact ? DFDesignSystem.Spacing.sm : DFDesignSystem.Spacing.md))

        if listStyle {
            rowContent
                .animation(.easeInOut(duration: 0.2), value: isSelected)
        } else {
            rowContent
                .background(DFDesignSystem.Colors.backgroundPrimary)
                .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius))
                .overlay(
                    RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius)
                        .stroke(
                            selectable && isSelected ? DFDesignSystem.Colors.brandPrimary : DFDesignSystem.Colors.divider.opacity(0.35),
                            lineWidth: selectable && isSelected ? 2.5 : 1
                        )
                )
                .dfShadow(DFDesignSystem.Shadows.card)
                .animation(.easeInOut(duration: 0.2), value: isSelected)
        }
    }

    @ViewBuilder
    private var selectButton: some View {
        if listStyle {
            if isSelected {
                FilledSelectedButton(action: onSelect)
            } else {
                OutlinedSelectButton(action: onSelect)
            }
        } else {
            cardStyleSelectButton
        }
    }

    private var cardStyleSelectButton: some View {
        Button(action: onSelect) {
            if compact {
                HStack(spacing: 4) {
                    if isSelected {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.caption)
                    }
                    Text(isSelected ? "Selected" : "Select")
                        .font(DFDesignSystem.Typography.caption)
                        .fontWeight(.medium)
                }
                .foregroundStyle(isSelected ? DFDesignSystem.Colors.brandPrimary : DFDesignSystem.Colors.textInverse)
                .padding(.horizontal, DFDesignSystem.Spacing.sm)
                .padding(.vertical, DFDesignSystem.Spacing.xs)
                .background(isSelected ? DFDesignSystem.Colors.brandPrimary.opacity(0.15) : DFDesignSystem.Colors.brandPrimary)
                .clipShape(Capsule())
            } else {
                Text("Select")
                    .font(DFDesignSystem.Typography.bodyBold)
                    .foregroundStyle(DFDesignSystem.Colors.textInverse)
                    .padding(.horizontal, DFDesignSystem.Spacing.lg)
                    .padding(.vertical, DFDesignSystem.Spacing.sm)
                    .background(DFDesignSystem.Colors.brandPrimary)
                    .clipShape(Capsule())
            }
        }
        .buttonStyle(.plain)
    }

    private var circularBadge: some View {
        ZStack {
            Circle()
                .stroke(DFDesignSystem.Colors.backgroundSecondary, lineWidth: ringLineWidth)
            Circle()
                .trim(from: 0, to: min(CGFloat(probability), 1.0))
                .stroke(ringColor(for: probability), style: StrokeStyle(lineWidth: ringLineWidth, lineCap: .round))
                .rotationEffect(.degrees(-90))
            Text("\(Int(probability * 100))%")
                .font(listStyle || compact ? .caption2 : DFDesignSystem.Typography.caption)
                .fontWeight(.semibold)
                .foregroundStyle(DFDesignSystem.Colors.textPrimary)
        }
        .frame(width: badgeSize, height: badgeSize)
    }
}

#Preview("Card row") {
    VStack(spacing: DFDesignSystem.Spacing.sm) {
        ProbabilityCardRow(
            displayName: "Dermatofibroma",
            subtitle: "",
            probability: 0.64,
            isSelected: false,
            onSelect: {}
        )
        ProbabilityCardRow(
            displayName: "Melanocytic Nevus",
            subtitle: "",
            probability: 0.16,
            isSelected: true,
            onSelect: {}
        )
    }
    .padding(DFDesignSystem.Spacing.md)
}

#Preview("Compact") {
    VStack(spacing: DFDesignSystem.Spacing.sm) {
        ProbabilityCardRow(
            displayName: "Melanocytic Nevus",
            subtitle: "",
            probability: 0.87,
            isSelected: true,
            onSelect: {},
            compact: true
        )
        ProbabilityCardRow(
            displayName: "Melanoma",
            subtitle: "",
            probability: 0.04,
            isSelected: false,
            onSelect: {},
            compact: true
        )
    }
    .padding(DFDesignSystem.Spacing.md)
}

#Preview("List style") {
    let items: [(String, Double, Bool)] = [("Melanocytic Nevus", 0.87, true), ("Melanoma", 0.04, false)]
    return VStack(spacing: 0) {
        ForEach(Array(items.enumerated()), id: \.offset) { index, item in
            if index > 0 {
                Divider().background(DFDesignSystem.Colors.divider)
            }
            ProbabilityCardRow(
                displayName: item.0,
                subtitle: "",
                probability: item.1,
                isSelected: item.2,
                onSelect: {},
                listStyle: true
            )
        }
    }
    .padding(DFDesignSystem.Spacing.md)
}
