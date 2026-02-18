//
//  LesionTypeDetailView.swift
//  DermFusion
//
//  Redesigned educational detail screen — static hero image with overlapping card.
//  Apple Maps / Apple Travel style. No toolbar buttons, no tap-to-expand.
//  Used by the Learn tab for educational lesion type content.
//

import SwiftUI

// MARK: - Lesion Type Detail (Learn Tab)

struct LesionTypeDetailView: View {

    let content: LesionEducation

    private let heroHeight: CGFloat = 380
    private let cardOverlap: CGFloat = 28
    private let cardRadius: CGFloat = 24

    var body: some View {
        ZStack(alignment: .topLeading) {
            DFDesignSystem.Colors.backgroundPrimary
                .ignoresSafeArea()

            ScrollView(.vertical, showsIndicators: false) {
                VStack(spacing: 0) {
                    heroSection
                    contentCard
                        .offset(y: -cardOverlap)
                        .ignoresSafeArea(edges: .bottom)
                }
                
            }
            .coordinateSpace(name: "scroll")
            .scrollContentBackground(.hidden)
            .ignoresSafeArea(edges: .bottom)
        }
        .toolbarBackground(.hidden, for: .navigationBar)
        .toolbarColorScheme(.dark, for: .navigationBar)
        .hideTabBarWhenPushed()
        .ignoresSafeArea(edges: [.top, .bottom])
    }
}

// MARK: - Hero Image

private extension LesionTypeDetailView {

    var heroSection: some View {
        GeometryReader { geo in
            let minY = geo.frame(in: .named("scroll")).minY
            let isStretching = minY > 0

            heroImage
                .frame(
                    width: geo.size.width,
                    height: isStretching ? heroHeight + minY : heroHeight
                )
                .clipped()
                .offset(y: isStretching ? -minY : 0)

            // Category badge floating on image
            if let category = content.category {
                categoryBadge(category)
                    .position(x: geo.size.width - 60, y: heroHeight - 44)
            }
        }
        .frame(height: heroHeight)
    }

    private var heroImageName: String? {
        content.examples.first?.imageName ?? content.category?.assetImageName
    }

    @ViewBuilder
    var heroImage: some View {
        if let imageName = heroImageName {
            Image(imageName)
                .resizable()
                .aspectRatio(contentMode: .fill)
        } else {
            ZStack {
                LinearGradient(
                    colors: [
                        DFDesignSystem.Colors.chartColor(for: content.category ?? .nevus).opacity(0.9),
                        DFDesignSystem.Colors.brandPrimary
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                Image(systemName: "photo.on.rectangle.angled")
                    .font(.system(size: 48, weight: .ultraLight))
                    .foregroundStyle(.white.opacity(0.5))
            }
        }
    }

    func categoryBadge(_ category: LesionCategory) -> some View {
        Text(content.severity)
            .font(.system(size: 12, weight: .bold, design: .rounded))
            .textCase(.uppercase)
            .tracking(0.5)
            .foregroundStyle(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(
                Capsule(style: .continuous)
                    .fill(severityColor.opacity(0.85))
                    .shadow(color: severityColor.opacity(0.3), radius: 6, y: 2)
            )
    }

    var severityColor: Color {
        switch content.severity.lowercased() {
        case "benign": return DFDesignSystem.Colors.riskLow
        case "malignant": return DFDesignSystem.Colors.riskHigh
        default: return DFDesignSystem.Colors.riskModerate
        }
    }
}

// MARK: - Content Card

private extension LesionTypeDetailView {

    var contentCard: some View {
        VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.md) {

            // ── Title + Metadata ──
            headerSection
            severityRow

            cardDivider

            // ── Overview ──
            overviewSection

            cardDivider

            // ── Key Visual Features ──
            visualFeaturesSection

            cardDivider

            // ── Dermoscopic Clues ──
            dermoscopySection

            cardDivider

            // ── Demographics ──
            demographicsSection

            cardDivider

            // ── References ──
            referencesSection

            // ── Disclaimer ──
            disclaimerSection

            Spacer().frame(height: DFDesignSystem.Spacing.lg)
        }
        .padding(.horizontal, DFDesignSystem.Spacing.lg)
        .padding(.top, cardOverlap + DFDesignSystem.Spacing.sm)
        .padding(.bottom, DFDesignSystem.Spacing.md)
        .background(
            RoundedRectangle(cornerRadius: cardRadius, style: .continuous)
                .fill(DFDesignSystem.Colors.cardBackground)
                .shadow(color: Color.black.opacity(0.08), radius: 20, x: 0, y: -8)
                
        )
    }

    var cardDivider: some View {
        Divider()
            .background(DFDesignSystem.Colors.textSecondary.opacity(0.12))
    }
}

// MARK: - Header

private extension LesionTypeDetailView {

    var headerSection: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(content.title)
                .font(DFDesignSystem.Typography.title2)
                .fontWeight(.bold)
                .foregroundStyle(DFDesignSystem.Colors.textPrimary)

            if let category = content.category {
                Text(category.rawValue.uppercased())
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .tracking(1)
                    .foregroundStyle(DFDesignSystem.Colors.textSecondary)
            }
        }
    }

    var severityRow: some View {
        HStack(spacing: DFDesignSystem.Spacing.sm) {
            metadataChip(icon: "stethoscope", label: content.severity)

            if !content.demographics.ageRange.isEmpty {
                metadataChip(icon: "person.crop.circle", label: content.demographics.ageRange)
            }

            Spacer()
        }
    }

    func metadataChip(icon: String, label: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: icon)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(DFDesignSystem.Colors.textSecondary)
            Text(label)
                .font(.system(size: 12, weight: .medium, design: .rounded))
                .foregroundStyle(DFDesignSystem.Colors.textSecondary)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(
            Capsule(style: .continuous)
                .fill(DFDesignSystem.Colors.backgroundSecondary.opacity(0.6))
        )
    }
}

// MARK: - Overview

private extension LesionTypeDetailView {

    var overviewSection: some View {
        VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.xs) {
            sectionHeader("Overview")

            Text(content.overview)
                .font(DFDesignSystem.Typography.body)
                .foregroundStyle(DFDesignSystem.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)
                .lineSpacing(2)

            if !content.prevalenceNote.isEmpty {
                Text(content.prevalenceNote)
                    .font(DFDesignSystem.Typography.caption)
                    .foregroundStyle(DFDesignSystem.Colors.textTertiary)
                    .padding(.top, 1)
            }
        }
    }
}

// MARK: - Visual Features

private extension LesionTypeDetailView {

    var visualFeaturesSection: some View {
        VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.xs) {
            sectionHeader("Key Visual Features")

            VStack(spacing: DFDesignSystem.Spacing.xs) {
                ForEach(Array(content.keyVisualFeatures.enumerated()), id: \.offset) { index, feature in
                    featureRow(icon: "eye", text: feature, index: index)
                }
            }
        }
    }
}

// MARK: - Dermoscopy

private extension LesionTypeDetailView {

    var dermoscopySection: some View {
        VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.xs) {
            sectionHeader("Dermoscopic Clues")

            VStack(spacing: DFDesignSystem.Spacing.xs) {
                ForEach(Array(content.dermoscopicClues.enumerated()), id: \.offset) { index, clue in
                    featureRow(icon: "magnifyingglass", text: clue, index: index)
                }
            }
        }
    }
}

// MARK: - Demographics

private extension LesionTypeDetailView {

    var demographicsSection: some View {
        VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.xs) {
            sectionHeader("Typical Demographics")

            VStack(spacing: DFDesignSystem.Spacing.xs) {
                demographicRow(icon: "calendar.circle", title: "Age Range", value: content.demographics.ageRange)
                demographicRow(icon: "person.2.circle", title: "Sex Tendency", value: content.demographics.sexTendency)
                demographicRow(
                    icon: "mappin.circle",
                    title: "Common Sites",
                    value: content.demographics.commonSites.joined(separator: ", ")
                )
            }
        }
    }

    func demographicRow(icon: String, title: String, value: String) -> some View {
        HStack(alignment: .top, spacing: DFDesignSystem.Spacing.xs) {
            Image(systemName: icon)
                .font(.system(size: 20, weight: .light))
                .foregroundStyle(DFDesignSystem.Colors.accent)
                .frame(width: 28, alignment: .center)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.system(size: 12, weight: .semibold, design: .rounded))
                    .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                    .textCase(.uppercase)
                    .tracking(0.3)

                Text(value)
                    .font(DFDesignSystem.Typography.body)
                    .foregroundStyle(DFDesignSystem.Colors.textPrimary)
            }

            Spacer()
        }
        .padding(DFDesignSystem.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(DFDesignSystem.Colors.backgroundSecondary.opacity(0.4))
        )
    }
}

// MARK: - References

private extension LesionTypeDetailView {

    var referencesSection: some View {
        VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.xs) {
            sectionHeader("References")

            VStack(spacing: DFDesignSystem.Spacing.xs) {
                ForEach(content.references) { reference in
                    HStack(alignment: .top, spacing: DFDesignSystem.Spacing.xs) {
                        Image(systemName: "doc.text")
                            .font(.system(size: 14, weight: .medium))
                            .foregroundStyle(DFDesignSystem.Colors.accent)
                            .frame(width: 24, alignment: .center)

                        VStack(alignment: .leading, spacing: 2) {
                            Text(reference.label)
                                .font(DFDesignSystem.Typography.subheadline)
                                .fontWeight(.medium)
                                .foregroundStyle(DFDesignSystem.Colors.textPrimary)

                            Text(reference.url)
                                .font(.system(size: 11))
                                .foregroundStyle(DFDesignSystem.Colors.accent.opacity(0.8))
                                .lineLimit(1)
                                .truncationMode(.middle)
                        }

                        Spacer()

                        Image(systemName: "arrow.up.right")
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(DFDesignSystem.Colors.textSecondary.opacity(0.5))
                    }
                    .padding(DFDesignSystem.Spacing.xs)
                    .background(
                        RoundedRectangle(cornerRadius: 10, style: .continuous)
                            .fill(DFDesignSystem.Colors.backgroundSecondary.opacity(0.3))
                    )
                }
            }
        }
    }
}

// MARK: - Disclaimer

private extension LesionTypeDetailView {

    var disclaimerSection: some View {
        HStack(alignment: .top, spacing: DFDesignSystem.Spacing.xs) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 14))
                .foregroundStyle(DFDesignSystem.Colors.riskModerate)

            Text("This information is educational and does not replace clinical diagnosis. Please consult a dermatologist for any concerns.")
                .font(.system(size: 12, weight: .regular))
                .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(DFDesignSystem.Spacing.sm)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(DFDesignSystem.Colors.riskModerate.opacity(0.08))
        )
        .overlay(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .strokeBorder(DFDesignSystem.Colors.riskModerate.opacity(0.2), lineWidth: 1)
        )
    }
}

// MARK: - Shared Components

private extension LesionTypeDetailView {

    func sectionHeader(_ title: String) -> some View {
        Text(title)
            .font(DFDesignSystem.Typography.headline)
            .fontWeight(.bold)
            .foregroundStyle(DFDesignSystem.Colors.textPrimary)
    }

    func featureRow(icon: String, text: String, index: Int) -> some View {
        HStack(alignment: .top, spacing: DFDesignSystem.Spacing.xs) {
            Image(systemName: icon)
                .font(.system(size: 14, weight: .semibold))
                .foregroundStyle(DFDesignSystem.Colors.accent)
                .frame(width: 28, height: 28)
                .background(
                    Circle()
                        .fill(DFDesignSystem.Colors.accent.opacity(0.1))
                )

            Text(text)
                .font(DFDesignSystem.Typography.body)
                .foregroundStyle(DFDesignSystem.Colors.textPrimary)
                .fixedSize(horizontal: false, vertical: true)

            Spacer(minLength: 0)
        }
        .padding(DFDesignSystem.Spacing.xs)
        .background(
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .fill(DFDesignSystem.Colors.backgroundSecondary.opacity(0.35))
        )
    }
}

// MARK: - Preview

#Preview("LesionTypeDetailView") {
    NavigationStack {
        LesionTypeDetailView(
            content: LesionEducation(
                categoryCode: "nv",
                title: "Melanocytic Nevus",
                severity: "Benign",
                overview: "Common benign melanocytic lesion arising from proliferation of melanocytes. These are among the most frequently encountered skin lesions in clinical dermatology.",
                prevalenceNote: "Most frequent class in the HAM10000 dataset.",
                keyVisualFeatures: ["Symmetric shape", "Uniform pigment network", "Regular borders"],
                dermoscopicClues: ["Reticular pattern", "Regular globules", "Homogeneous pigmentation"],
                demographics: DemographicsNote(
                    ageRange: "20-60 years",
                    sexTendency: "No strong bias",
                    commonSites: ["Back", "Trunk", "Extremities"]
                ),
                examples: [],
                references: [
                    EducationReference(id: "r1", label: "ISIC Archive", url: "https://www.isic-archive.com")
                ]
            )
        )
    }
}
