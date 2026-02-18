//
//  DermoscopyBasicsView.swift
//  DermFusion
//
//  Educational screen introducing dermoscopy fundamentals. Native grouped list.
//

import SwiftUI

struct DermoscopyBasicsView: View {

    let paragraphs: [EducationParagraph]

    var body: some View {
        List {
            ForEach(paragraphs) { paragraph in
                Section {
                    Text(paragraph.body)
                        .font(.body)
                        .foregroundStyle(.primary)
                        .padding(.vertical, 4)
                } header: {
                    Text(paragraph.title)
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Dermoscopy Basics")
        .navigationBarTitleDisplayMode(.inline)
        .hideTabBarWhenPushed()
    }
}

#Preview {
    NavigationStack {
        DermoscopyBasicsView(
            paragraphs: [
                EducationParagraph(
                    id: "demo",
                    title: "What is Dermoscopy?",
                    body: "Dermoscopy uses magnification and controlled illumination to examine skin lesions. It helps clinicians distinguish structures and patterns that inform classification."
                )
            ]
        )
    }
}

#Preview("Dark") {
    NavigationStack {
        DermoscopyBasicsView(
            paragraphs: [
                EducationParagraph(
                    id: "demo",
                    title: "What is Dermoscopy?",
                    body: "Dermoscopy uses magnification and controlled illumination."
                )
            ]
        )
    }
    .preferredColorScheme(.dark)
}
