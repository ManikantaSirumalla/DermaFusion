//
//  MetadataInputViewModel.swift
//  DermFusion
//
//  Validates metadata and maps user input into the model-ready structure.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Combine
import Foundation

@MainActor
final class MetadataInputViewModel: ObservableObject {

    // MARK: - Properties

    @Published var age: Int?
    @Published var sex: Sex = .unspecified
    @Published var bodyRegion: BodyRegion?

    // MARK: - Public API

    var canAnalyze: Bool {
        bodyRegion != nil
    }
}
