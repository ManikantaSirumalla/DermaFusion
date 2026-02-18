//
//  MetadataInputView.swift
//  DermFusion
//
//  Clinical metadata capture before analysis. Age and gender required; name and ethnicity optional.
//

import SwiftUI

/// Captures age, sex, lesion location, and optional name and ethnicity before analysis.
struct MetadataInputView: View {

    @StateObject private var viewModel: ScanViewModel
    @FocusState private var focusedField: Field?
    @State private var showAnalysis = false

    init(viewModel: ScanViewModel) {
        _viewModel = StateObject(wrappedValue: viewModel)
    }

    enum Field {
        case name
    }

    var body: some View {
        List {
            Section {
                TextField("Name (optional)", text: $viewModel.name)
                    .focused($focusedField, equals: .name)
                    .textContentType(.name)
                    .autocorrectionDisabled(false)

                Picker("Ethnicity (optional)", selection: $viewModel.selectedEthnicity) {
                    ForEach(Ethnicity.allCases) { option in
                        Text(option.displayName).tag(option)
                    }
                }

                DFSlider(title: "Age", value: $viewModel.age, range: 0 ... 100)

                VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.sm) {
                    Text("Gender")
                        .font(DFDesignSystem.Typography.subheadline)
                    DFSegmentedControl(
                        title: "Gender",
                        options: Sex.allCases,
                        label: { $0.displayName },
                        selection: $viewModel.sex
                    )
                }
            } header: {
                Text("Patient Metadata")
            } footer: {
                Text("Age and gender are required. Name and ethnicity are optional.")
            }

            Section {
                NavigationLink {
                    BodyMapView(sex: viewModel.sex, selectedRegion: $viewModel.selectedRegion)
                } label: {
                    Label(
                        viewModel.selectedRegion?.displayName ?? "Select Location",
                        systemImage: DFDesignSystem.Icons.bodyMap
                    )
                }
            } header: {
                Text("Lesion Location")
            } footer: {
                Text("Required to continue.")
            }

            Section {
                Button {
                    showAnalysis = true
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: "magnifyingglass")
                            .font(.title3.weight(.medium))
                        Text("Analyze")
                            .font(.title3.weight(.semibold))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(
                        Capsule()
                            .fill(viewModel.canAnalyze ? DFDesignSystem.Colors.brandPrimary : Color(.tertiarySystemFill))
                    )
                    .foregroundStyle(viewModel.canAnalyze ? Color(.systemBackground) : Color(.tertiaryLabel))
                }
                .buttonStyle(.plain)
                .disabled(!viewModel.canAnalyze)
                .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
                .listRowBackground(Color.clear)
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Metadata")
        .navigationBarTitleDisplayMode(.inline)
        .hideTabBarWhenPushed()
        .navigationDestination(isPresented: $showAnalysis) {
            AnalysisView(viewModel: viewModel)
        }
    }
}

#Preview("Default") {
    NavigationStack {
        MetadataInputView(viewModel: ScanViewModel())
    }
}

#Preview("Dark") {
    NavigationStack {
        MetadataInputView(viewModel: ScanViewModel())
    }
    .preferredColorScheme(.dark)
}
