//
//  AppDataStore.swift
//  DermFusion
//
//  Shared app-level state for tab routing and locally saved scan records.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Combine
import Foundation

@MainActor
final class AppDataStore: ObservableObject {

    // MARK: - Published State

    @Published var selectedTab: AppTab = .scan
    @Published private(set) var records: [ScanRecord] = []
    /// When true, the scan tab should pop to ScanHomeView (e.g. after saving a scan).
    @Published var shouldDismissScanFlow = false

    // MARK: - Computed Properties

    var latestRecord: ScanRecord? {
        records.first
    }

    // MARK: - Public API

    func save(result: DiagnosisResult, metadata: MetadataInput) {
        let record = ScanRecord(
            imageData: Data(),
            age: metadata.age,
            sex: metadata.sex.displayName,
            lesionLocation: metadata.bodyRegion.displayName,
            predictions: result.probabilities.reduce(into: [:]) { partial, item in
                partial[item.key.rawValue] = item.value
            },
            primaryDiagnosis: result.primaryDiagnosis.displayName,
            riskLevel: result.riskLevel.rawValue,
            gradcamImageData: nil,
            userNotes: nil
        )
        records.insert(record, at: 0)
    }

    func delete(_ record: ScanRecord) {
        records.removeAll { $0.id == record.id }
    }

    func clearAll() {
        records.removeAll()
    }
}
