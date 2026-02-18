//
//  BodyMapViewModel.swift
//  DermFusion
//
//  View model for body-region selection and display state.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Combine
import Foundation

@MainActor
final class BodyMapViewModel: ObservableObject {

    // MARK: - Properties

    @Published var selectedRegion: BodyRegion?

    var selectedRegionLabel: String {
        selectedRegion?.displayName ?? "No Region Selected"
    }
}
