//
//  DFError.swift
//  DermFusion
//
//  Unified error model for DermaFusion features.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation
import Combine

/// App-wide failures with user-facing context.
enum DFError: Error, Identifiable, LocalizedError {
    case generic(title: String, message: String)

    var id: String { localizedDescription }

    var errorDescription: String? {
        switch self {
        case .generic(let title, let message):
            return "\(title): \(message)"
        }
    }
}
