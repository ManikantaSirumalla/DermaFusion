//
//  Sex.swift
//  DermFusion
//
//  Sex input options for metadata capture.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

/// Supported sex inputs aligned with metadata encoding policy.
enum Sex: String, CaseIterable, Identifiable, Sendable {
    case male
    case female
    case unspecified

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .male:
            return "Male"
        case .female:
            return "Female"
        case .unspecified:
            return "Prefer not to say"
        }
    }
}
