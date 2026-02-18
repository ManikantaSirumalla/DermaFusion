//
//  PersistenceService.swift
//  DermFusion
//
//  Local persistence service for scan records using on-device storage.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

/// Handles create/read/delete operations for scan history records.
protocol PersistenceServiceProtocol: Sendable {
    func save(_ record: ScanRecord) async throws
    func fetchAll() async throws -> [ScanRecord]
}

/// In-memory placeholder implementation used until SwiftData wiring is complete.
actor PersistenceService: PersistenceServiceProtocol {

    // MARK: - Properties

    private var records: [ScanRecord] = []

    // MARK: - Public API

    func save(_ record: ScanRecord) async throws {
        records.append(record)
    }

    func fetchAll() async throws -> [ScanRecord] {
        records.sorted { $0.timestamp > $1.timestamp }
    }
}
