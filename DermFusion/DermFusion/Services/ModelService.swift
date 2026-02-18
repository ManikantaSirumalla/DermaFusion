//
//  ModelService.swift
//  DermFusion
//
//  CoreML inference service scaffold for DermaFusion model execution.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

/// Runs model inference on preprocessed image and metadata inputs.
protocol ModelServiceProtocol: Sendable {
    func classify() async throws -> DiagnosisResult
}

/// Placeholder model service until CoreML integration is wired.
actor ModelService: ModelServiceProtocol {
    func classify() async throws -> DiagnosisResult {
        DiagnosisResult.sample
    }
}
