//
//  ResultsView.swift
//  DermFusion
//
//  Analysis results: native grouped list with hero, assessment, probabilities, actions.
//

import SwiftUI

/// Displays the analysis output with risk context and educational framing.
struct ResultsView: View {

    let result: DiagnosisResult
    let metadata: MetadataInput?
    let onSave: () -> Void
    let onLearnMore: () -> Void

    @State private var showGradCAM = false
    @State private var didSave = false
    @State private var showSavedToast = false
    @State private var isAssessmentExpanded = false
    @EnvironmentObject private var appDataStore: AppDataStore
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    init(
        result: DiagnosisResult,
        metadata: MetadataInput? = nil,
        onSave: @escaping () -> Void = {},
        onLearnMore: @escaping () -> Void = {}
    ) {
        self.result = result
        self.metadata = metadata
        self.onSave = onSave
        self.onLearnMore = onLearnMore
    }

    private var preResultRows: [(category: LesionCategory, probability: Double)] {
        LesionCategory.allCases.map { category in
            (category, result.probabilities[category] ?? 0)
        }.sorted { $0.probability > $1.probability }
    }

    private func metadataRow(icon: String, title: String, value: String) -> some View {
        HStack(alignment: .top, spacing: DFDesignSystem.Spacing.sm) {
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
                    .fontWeight(.medium)
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

    private var compactAssessmentCard: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.25)) {
                isAssessmentExpanded = true
            }
        } label: {
            HStack(spacing: 12) {
                Image("raw")
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 56, height: 56)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

                VStack(alignment: .leading, spacing: 2) {
                    Text(result.primaryDiagnosis.displayName)
                        .font(.body)
                        .fontWeight(.medium)
                    Text("\(Int(result.topProbability * 1000) / 10)% confidence")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                DFRiskBadge(level: result.riskLevel, compact: true)
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
        .listRowInsets(EdgeInsets(top: 12, leading: 16, bottom: 12, trailing: 16))
    }

    private var expandedAssessmentCard: some View {
        VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.md) {
            GradCAMOverlayView(showGradCAM: $showGradCAM)
                .frame(maxWidth: .infinity)
                .frame(height: 300)
                .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius))
            Toggle("Show GradCAM", isOn: $showGradCAM)
                .tint(DFDesignSystem.Colors.brandPrimary)

            Text(result.primaryDiagnosis.displayName)
                .font(DFDesignSystem.Typography.title2)
                .fontWeight(.bold)
                .foregroundStyle(DFDesignSystem.Colors.textPrimary)

            Text("\(Int(result.topProbability * 1000) / 10)% confidence")
                .font(DFDesignSystem.Typography.body)
                .foregroundStyle(DFDesignSystem.Colors.textSecondary)

            HStack {
                DFRiskBadge(level: result.riskLevel)
                Spacer()
            }

            HStack {
                Button {
                    onLearnMore()
                } label: {
                    HStack(spacing: 4) {
                        Text("Learn More")
                            .font(DFDesignSystem.Typography.body)
                            .fontWeight(.medium)
                        Image(systemName: "arrow.right")
                            .font(.caption.weight(.semibold))
                    }
                    .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                }
                .buttonStyle(.plain)

                Spacer()

                Button {
                    withAnimation(.easeInOut(duration: 0.25)) {
                        isAssessmentExpanded = false
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.up.circle.fill")
                            .font(.title3)
                        Text("Show less")
                            .font(DFDesignSystem.Typography.caption)
                            .fontWeight(.medium)
                    }
                    .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                }
                .buttonStyle(.plain)
            }
        }
        .listRowInsets(EdgeInsets(top: 12, leading: 16, bottom: 12, trailing: 16))
    }

    var body: some View {
        List {
            Section {
                if isAssessmentExpanded {
                    expandedAssessmentCard
                } else {
                    compactAssessmentCard
                }
            } header: {
                Text("Assessment")
            }

            Section {
                ProbabilityChartView(probabilities: preResultRows)
                    .padding(.vertical, 8)
                    .listRowInsets(EdgeInsets(top: 12, leading: 16, bottom: 12, trailing: 16))
            } header: {
                Text("Probabilities")
            }

            Section {
                if let metadata {
                    VStack(spacing: 0) {
                        metadataRow(
                            icon: "calendar.circle",
                            title: "Age",
                            value: metadata.age.map(String.init) ?? "Not provided"
                        )

                        Divider()
                            .background(DFDesignSystem.Colors.divider)
                            .padding(.vertical, DFDesignSystem.Spacing.xs)

                        metadataRow(
                            icon: "person.2.circle",
                            title: "Sex",
                            value: metadata.sex.displayName
                        )

                        Divider()
                            .background(DFDesignSystem.Colors.divider)
                            .padding(.vertical, DFDesignSystem.Spacing.xs)

                        metadataRow(
                            icon: "mappin.circle",
                            title: "Location",
                            value: metadata.bodyRegion.displayName
                        )

                        if let name = metadata.name, !name.isEmpty {
                            Divider()
                                .background(DFDesignSystem.Colors.divider)
                                .padding(.vertical, DFDesignSystem.Spacing.xs)
                            metadataRow(icon: "person.crop.circle", title: "Name", value: name)
                        }
                        if let ethnicity = metadata.ethnicity, !ethnicity.isEmpty {
                            Divider()
                                .background(DFDesignSystem.Colors.divider)
                                .padding(.vertical, DFDesignSystem.Spacing.xs)
                            metadataRow(icon: "globe", title: "Ethnicity", value: ethnicity)
                        }
                    }
                } else {
                    Text("Metadata unavailable for this classification result.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Metadata")
            }

            Section {
                Button {
                    onSave()
                    didSave = true
                    showSavedToast = true
                    Task { @MainActor in
                        try? await Task.sleep(nanoseconds: 2_000_000_000)
                        showSavedToast = false
                        appDataStore.shouldDismissScanFlow = true
                    }
                } label: {
                    HStack {
                        Spacer()
                        Text(didSave ? "Saved ✓" : "Save Scan")
                            .font(DFDesignSystem.Typography.body)
                            .fontWeight(.semibold)
                        Spacer()
                    }
                    .padding(.vertical, 14)
                    .foregroundStyle(DFDesignSystem.Colors.textInverse)
                    .background(didSave ? Color.green : DFDesignSystem.Colors.brandPrimary)
                    .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius, style: .continuous))
                }
                .disabled(didSave)
                .buttonStyle(.plain)
                .listRowInsets(EdgeInsets(top: 0, leading: 0, bottom: 0, trailing: 0))
                .listRowBackground(Color.clear)
            }

            Section {
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: DFDesignSystem.Icons.disclaimer)
                        .font(.body)
                        .foregroundStyle(DFDesignSystem.Colors.riskModerate)
                        .symbolRenderingMode(.hierarchical)
                    Text("Research tool only — not a medical diagnosis. Always consult a dermatologist.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Results")
        .navigationBarTitleDisplayMode(.inline)
        .hideTabBarWhenPushed()
        .overlay(alignment: .bottom) {
            if showSavedToast {
                savedToast
            }
        }
        .animation(.easeInOut(duration: 0.25), value: showSavedToast)
    }

    private var savedToast: some View {
        HStack(spacing: 10) {
            Image(systemName: "checkmark.circle.fill")
                .font(.title3)
                .foregroundStyle(.white)
            Text("Scan has been successfully saved to history")
                .font(.subheadline.weight(.medium))
                .foregroundStyle(.white)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
        .background(
            Capsule(style: .continuous)
                .fill(.black.opacity(0.78))
        )
        .padding(.horizontal, 24)
        .padding(.bottom, 40)
    }
}

#Preview("Low Risk") {
    NavigationStack {
        ResultsView(result: .sample, metadata: MetadataInput(age: 55, sex: .male, bodyRegion: .back))
            .environmentObject(AppDataStore())
    }
}

#Preview("Dark") {
    NavigationStack {
        ResultsView(result: .sample, metadata: MetadataInput(age: nil, sex: .unspecified, bodyRegion: .face))
            .environmentObject(AppDataStore())
    }
    .preferredColorScheme(.dark)
}
