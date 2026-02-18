//
//  AboutDisclaimerView.swift
//  DermFusion
//
//  Full medical disclaimer. Native list with section footer.
//

import SwiftUI

struct AboutDisclaimerView: View {

    var body: some View {
        List {
            Section {
                HStack(alignment: .top, spacing: 12) {
                    Image(systemName: DFDesignSystem.Icons.disclaimer)
                        .font(.title2)
                        .foregroundStyle(DFDesignSystem.Colors.riskModerate)
                        .symbolRenderingMode(.hierarchical)
                    Text(DFMedicalDisclaimer.fullText)
                        .font(.body)
                        .foregroundStyle(.primary)
                }
                .padding(.vertical, 8)
            } header: {
                Text("Medical Disclaimer")
            } footer: {
                Text("Always consult a qualified dermatologist for skin concerns.")
                    .font(.caption)
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Medical Disclaimer")
        .navigationBarTitleDisplayMode(.inline)
        .hideTabBarWhenPushed()
    }
}

#Preview {
    NavigationStack { AboutDisclaimerView() }
}
