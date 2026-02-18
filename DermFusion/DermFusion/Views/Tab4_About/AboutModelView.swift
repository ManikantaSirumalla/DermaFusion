//
//  AboutModelView.swift
//  DermFusion
//
//  Model architecture details and constraints. Native grouped list.
//

import SwiftUI

struct AboutModelView: View {

    var body: some View {
        List {
            Section {
                Text("EfficientNet-B4 image encoder with metadata fusion for 7-class lesion classification (mel, nv, bcc, akiec, bkl, df, vasc).")
                    .font(.body)
                    .foregroundStyle(.primary)
                    .padding(.vertical, 4)
            } header: {
                Text("Architecture")
            }

            Section {
                listBullet("Educational output only; not a diagnosis.")
                listBullet("Image quality and lighting may affect classification.")
                listBullet("Clinical confirmation by a dermatologist is recommended.")
            } header: {
                Text("Limitations")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Model Information")
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
    NavigationStack { AboutModelView() }
}
