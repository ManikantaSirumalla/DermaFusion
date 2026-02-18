//
//  ABCDERuleView.swift
//  DermFusion
//
//  ABCDE mnemonic educational content. Native grouped list.
//

import SwiftUI

/// Presents ABCDE mnemonic educational content.
struct ABCDERuleView: View {

    let points: [ABCDEPoint]

    var body: some View {
        List {
            Section {
                Text("The ABCDE rule is a simple guide to the typical warning signs of melanoma and other suspicious lesions. Use it only as an educational reference; it does not replace a clinical exam.")
                    .font(.body)
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 4)
            } header: {
                Text("About the ABCDE Rule")
            }

            ForEach(points) { point in
                Section {
                    Text(point.description)
                        .font(.body)
                        .foregroundStyle(.primary)
                        .padding(.vertical, 4)
                } header: {
                    Text("\(point.letter) — \(point.title)")
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("ABCDE Rule")
        .navigationBarTitleDisplayMode(.inline)
        .hideTabBarWhenPushed()
    }
}

#Preview {
    NavigationStack {
        ABCDERuleView(
            points: [
                ABCDEPoint(letter: "A", title: "Asymmetry", description: "One half of the lesion does not match the other half."),
                ABCDEPoint(letter: "B", title: "Border", description: "Edges are irregular, ragged, or blurred.")
            ]
        )
    }
}
