//
//  AboutReferencesView.swift
//  DermFusion
//
//  Educational references and source acknowledgements. Native list rows.
//

import SwiftUI

struct AboutReferencesView: View {

    private let references: [(title: String, url: String, description: String)] = [
        (
            "ISIC Archive",
            "https://api.isic-archive.com",
            "Image and metadata source for educational examples and model-aligned references."
        ),
        (
            "Dermoscopedia",
            "https://dermoscopedia.org",
            "Clinical dermoscopy terminology and educational reference content."
        )
    ]

    var body: some View {
        List {
            Section {
                ForEach(references, id: \.title) { ref in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(ref.title)
                            .font(.body)
                            .fontWeight(.medium)
                        Text(ref.url)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.middle)
                        Text(ref.description)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }
            } header: {
                Text("Sources")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("References")
        .navigationBarTitleDisplayMode(.inline)
        .hideTabBarWhenPushed()
    }
}

#Preview {
    NavigationStack { AboutReferencesView() }
}
