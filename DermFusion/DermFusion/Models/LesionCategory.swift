//
//  LesionCategory.swift
//  DermFusion
//
//  Model output category definitions for DermaFusion classification.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

/// Supported lesion categories from the 7-class training taxonomy.
enum LesionCategory: String, CaseIterable, Sendable {
    case melanoma = "mel"
    case nevus = "nv"
    case bcc = "bcc"
    case akiec = "akiec"
    case bkl = "bkl"
    case dermatofibroma = "df"
    case vascular = "vasc"

    /// Full display name (same as displayName; for detail view compatibility).
    var fullName: String { displayName }

    var displayName: String {
        switch self {
        case .melanoma:
            return "Melanoma"
        case .nevus:
            return "Melanocytic Nevus"
        case .bcc:
            return "Basal Cell Carcinoma"
        case .akiec:
            return "Actinic Keratosis"
        case .bkl:
            return "Benign Keratosis"
        case .dermatofibroma:
            return "Dermatofibroma"
        case .vascular:
            return "Vascular Lesion"
        }
    }

    /// Asset catalog image name for this class (Learn section detail hero image).
    var assetImageName: String {
        switch self {
        case .melanoma: return "melanoma"
        case .nevus: return "benign melanocytic nevus"
        case .bcc: return "basal cell carcinoma"
        case .akiec: return "actinic keratosis"
        case .bkl: return "benign keratosis"
        case .dermatofibroma: return "dermatofibroma"
        case .vascular: return "vascular lesion"
        }
    }
}
