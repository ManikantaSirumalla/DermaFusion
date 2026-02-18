//
//  ScanDetailView.swift
//  DermFusion
//
//  Detail screen for a saved scan record. Uses same compact/expandable card as ResultsView.
//

import SwiftUI
import UIKit

/// Displays a saved scan record with assessment card and probabilities.
struct ScanDetailView: View {

    let record: ScanRecord
    @State private var showGradCAM = false
    @State private var isAssessmentExpanded = false
    @State private var selectedClass: LesionCategory?
    /// When non-nil, the share sheet is presented with this PDF URL. Cleared on dismiss.
    @State private var shareablePDFItem: ShareablePDFItem?
    @State private var exportError: DFError?

    private var topProbability: Double {
        record.predictions.values.max() ?? 0
    }

    private var gaugeConfidence: Double {
        selectedClass.flatMap { record.probability(for: $0) } ?? topProbability
    }

    private var probabilityRows: [(category: LesionCategory, probability: Double)] {
        LesionCategory.allCases.map { category in
            (category, record.probability(for: category))
        }.sorted { $0.probability > $1.probability }
    }

    private func riskLevel(from raw: String) -> RiskLevel {
        RiskLevel(rawValue: raw) ?? .moderate
    }

    private func exportPDF() {
        do {
            let url = try PDFExportService().generatePDF(record: record)
            shareablePDFItem = ShareablePDFItem(url: url)
        } catch {
            exportError = DFError.generic(
                title: "Export Failed",
                message: error.localizedDescription
            )
        }
    }

    /// Settings-style list of 7 classifiers with thin dividers; no cards, no row-level selection border.
    private var probabilitiesListContent: some View {
        VStack(spacing: 0) {
            ForEach(Array(probabilityRows.enumerated()), id: \.element.category) { index, item in
                if index > 0 {
                    Divider()
                        .background(DFDesignSystem.Colors.divider)
                }
                ProbabilityCardRow(
                    displayName: item.category.displayName,
                    subtitle: "",
                    probability: item.probability,
                    isSelected: selectedClass == item.category,
                    onSelect: {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            selectedClass = item.category
                        }
                    },
                    listStyle: true
                )
            }
        }
    }

    private var compactAssessmentCard: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.25)) {
                isAssessmentExpanded = true
            }
        } label: {
            HStack(spacing: 12) {
                thumbnailView
                    .frame(width: 56, height: 56)
                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))

                VStack(alignment: .leading, spacing: 2) {
                    Text(record.primaryDiagnosis)
                        .font(.body)
                        .fontWeight(.medium)
                    Text("\(Int(topProbability * 1000) / 10)% confidence")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                DFRiskBadge(level: riskLevel(from: record.riskLevel), compact: true)
            }
            .padding(.vertical, 4)
        }
        .buttonStyle(.plain)
        .listRowInsets(EdgeInsets(top: 12, leading: 16, bottom: 12, trailing: 16))
    }

    @ViewBuilder
    private var thumbnailView: some View {
        if let uiImage = UIImage(data: record.imageData), !record.imageData.isEmpty {
            Image(uiImage: uiImage)
                .resizable()
                .aspectRatio(contentMode: .fill)
        } else {
            Image("raw")
                .resizable()
                .aspectRatio(contentMode: .fill)
        }
    }

    private var expandedAssessmentCard: some View {
        VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.md) {
            GradCAMOverlayView(showGradCAM: $showGradCAM)
                .frame(maxWidth: .infinity)
                .frame(height: 300)
                .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius))
            Toggle("Show GradCAM", isOn: $showGradCAM)
                .tint(DFDesignSystem.Colors.brandPrimary)

            Text(record.primaryDiagnosis)
                .font(DFDesignSystem.Typography.title2)
                .fontWeight(.bold)
                .foregroundStyle(DFDesignSystem.Colors.textPrimary)

            Text("\(Int(topProbability * 1000) / 10)% confidence")
                .font(DFDesignSystem.Typography.body)
                .foregroundStyle(DFDesignSystem.Colors.textSecondary)

            HStack {
                DFRiskBadge(level: riskLevel(from: record.riskLevel))
                Spacer()
            }

            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Scanned \(record.timestamp.formatted(date: .abbreviated, time: .shortened))")
                        .font(DFDesignSystem.Typography.caption)
                        .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                    Text("Location: \(record.lesionLocation)")
                        .font(DFDesignSystem.Typography.caption)
                        .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                }
                Spacer(minLength: 0)
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
                VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.lg) {
                    PreResultGaugeView(confidence: gaugeConfidence)
                        .frame(maxWidth: .infinity)
                        .padding(.top, DFDesignSystem.Spacing.lg)
                        .padding(.bottom, DFDesignSystem.Spacing.sm)

                    Text("Tap a class below to see its confidence on the gauge. The model outputs a probability for each of the 7 lesion types.")
                        .font(DFDesignSystem.Typography.caption)
                        .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                        .fixedSize(horizontal: false, vertical: true)

                    probabilitiesListContent
                }
                .listRowInsets(EdgeInsets(top: 12, leading: 16, bottom: 12, trailing: 16))
            } header: {
                Text("Probabilities")
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
        .navigationTitle("Scan Detail")
        .navigationBarTitleDisplayMode(.inline)
        .hideTabBarWhenPushed()
        .toolbar {
            ToolbarItem(placement: .topBarTrailing) {
                Menu {
                    Button {
                        exportPDF()
                    } label: {
                        Label("Export PDF", systemImage: DFDesignSystem.Icons.export)
                    }
                } label: {
                    Image(systemName: "ellipsis.circle")
                }
            }
        }
        .sheet(item: $shareablePDFItem) { item in
            ShareSheet(activityItems: [item.url])
                .onDisappear {
                    try? FileManager.default.removeItem(at: item.url)
                }
        }
        .alert(item: $exportError) { err in
            Alert(
                title: Text("Export Failed"),
                message: Text(err.localizedDescription),
                dismissButton: .default(Text("OK")) { exportError = nil }
            )
        }
        .onAppear {
            if selectedClass == nil {
                selectedClass = record.primaryDiagnosisLesionCategory
            }
        }
    }
}

// MARK: - Share sheet item

private struct ShareablePDFItem: Identifiable {
    let id = UUID()
    let url: URL
}

#Preview {
    let record = ScanRecord(
        imageData: Data(),
        age: 55,
        sex: "Male",
        lesionLocation: "Back",
        predictions: ["nv": 0.87, "mel": 0.04],
        primaryDiagnosis: "Melanocytic Nevus",
        riskLevel: "low"
    )
    return NavigationStack { ScanDetailView(record: record) }
}

#Preview("Dark") {
    let record = ScanRecord(
        imageData: Data(),
        age: 55,
        sex: "Male",
        lesionLocation: "Back",
        predictions: ["nv": 0.87, "mel": 0.04],
        primaryDiagnosis: "Melanocytic Nevus",
        riskLevel: "low"
    )
    return NavigationStack { ScanDetailView(record: record) }
        .preferredColorScheme(.dark)
}
