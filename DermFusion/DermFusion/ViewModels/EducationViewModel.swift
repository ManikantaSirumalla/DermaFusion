//
//  EducationViewModel.swift
//  DermFusion
//
//  Loads educational content for learn-tab screens from bundled resources.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Combine
import Foundation

@MainActor
final class EducationViewModel: ObservableObject {

    // MARK: - Properties

    @Published private(set) var content: EducationContentDocument = .fallback
    @Published private(set) var loadError: String?

    // MARK: - Initialization

    init() {
        loadBundledContent()
    }

    // MARK: - Derived Data

    var abcdePoints: [ABCDEPoint] {
        content.abcde
    }

    var dermoscopyParagraphs: [EducationParagraph] {
        content.dermoscopyBasics
    }

    var lesions: [LesionEducation] {
        content.lesions.sorted { lhs, rhs in
            lesionSortKey(for: lhs.categoryCode) < lesionSortKey(for: rhs.categoryCode)
        }
    }

    func lesion(for category: LesionCategory) -> LesionEducation? {
        content.lesions.first { $0.categoryCode == category.rawValue }
    }

    // MARK: - Private Helpers

    private func loadBundledContent() {
        guard let url = Bundle.main.url(forResource: "EducationalContent", withExtension: "json") else {
            loadError = "EducationalContent.json missing from bundle."
            content = .fallback
            return
        }

        do {
            let data = try Data(contentsOf: url)
            content = try JSONDecoder().decode(EducationContentDocument.self, from: data)
            loadError = nil
        } catch {
            loadError = "Unable to decode bundled educational content."
            content = .fallback
        }
    }

    private func lesionSortKey(for categoryCode: String) -> Int {
        let order: [String] = LesionCategory.allCases.map(\.rawValue)
        return order.firstIndex(of: categoryCode) ?? order.count
    }
}

private extension EducationContentDocument {
    static let fallback = EducationContentDocument(
        version: "fallback",
        lastUpdated: "n/a",
        sources: [],
        abcde: [
            ABCDEPoint(letter: "A", title: "Asymmetry", description: "One half of a lesion does not match the other half."),
            ABCDEPoint(letter: "B", title: "Border", description: "Edges are irregular, blurred, or uneven."),
            ABCDEPoint(letter: "C", title: "Color", description: "Multiple colors or uneven tone appear within one lesion."),
            ABCDEPoint(letter: "D", title: "Diameter", description: "A larger lesion may warrant closer review."),
            ABCDEPoint(letter: "E", title: "Evolution", description: "Recent change in size, shape, color, or symptoms.")
        ],
        dermoscopyBasics: [
            EducationParagraph(
                id: "fallback-derm-1",
                title: "Dermoscopy",
                body: "Dermoscopy magnifies skin structures that are often not visible to the naked eye."
            )
        ],
        lesions: []
    )
}
