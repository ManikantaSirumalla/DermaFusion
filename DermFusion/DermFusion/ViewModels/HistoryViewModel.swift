//
//  HistoryViewModel.swift
//  DermFusion
//
//  View model for loading, deleting, and empty-state handling of scan history.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Combine
import Foundation

@MainActor
final class HistoryViewModel: ObservableObject {

    // MARK: - Properties

    @Published private(set) var records: [ScanRecord] = []

    // MARK: - Public API

    var isEmpty: Bool {
        records.isEmpty
    }
}
