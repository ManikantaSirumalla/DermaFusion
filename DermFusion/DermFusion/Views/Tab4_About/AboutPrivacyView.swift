//
//  AboutPrivacyView.swift
//  DermFusion
//
//  Privacy and local-data policy. Native grouped list.
//

import SwiftUI

struct AboutPrivacyView: View {

    var body: some View {
        List {
            Section {
                Label("All analysis runs on your device. DermaFusion does not transmit health data.", systemImage: DFDesignSystem.Icons.aboutPrivacy)
                    .font(.body)
                    .foregroundStyle(.primary)
                    .symbolRenderingMode(.hierarchical)
            } header: {
                Text("Your Privacy")
            }

            Section {
                listBullet("No analytics SDKs for health content.")
                listBullet("No background uploads.")
                HStack {
                    Text("App Store")
                        .font(.body)
                    Spacer()
                    Text("Data Not Collected")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                }
            } header: {
                Text("Data Handling")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Privacy & Data")
        .navigationBarTitleDisplayMode(.inline)
        .hideTabBarWhenPushed()
    }

    private func listBullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•")
                .foregroundStyle(.secondary)
            Text(text)
                .font(.body)
                .foregroundStyle(.primary)
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    NavigationStack { AboutPrivacyView() }
}
