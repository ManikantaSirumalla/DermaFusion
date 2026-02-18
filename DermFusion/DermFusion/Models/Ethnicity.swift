//
//  Ethnicity.swift
//  DermFusion
//
//  Optional patient ethnicity options for metadata.
//

import Foundation

/// Optional ethnicity options for patient metadata. Stored only when user selects a value other than preferNotToSay.
enum Ethnicity: String, CaseIterable, Identifiable, Sendable {

    case asian
    case blackAfricanDescent
    case hispanicLatino
    case middleEasternNorthAfrican
    case nativeAmericanAlaskaNative
    case nativeHawaiianPacificIslander
    case whiteCaucasian
    case mixedMultiracial
    case other
    case preferNotToSay

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .preferNotToSay: return "Prefer not to say"
        case .asian: return "Asian"
        case .blackAfricanDescent: return "Black / African descent"
        case .hispanicLatino: return "Hispanic / Latino"
        case .middleEasternNorthAfrican: return "Middle Eastern / North African"
        case .nativeAmericanAlaskaNative: return "Native American / Alaska Native"
        case .nativeHawaiianPacificIslander: return "Native Hawaiian / Pacific Islander"
        case .whiteCaucasian: return "White / Caucasian"
        case .mixedMultiracial: return "Mixed / Multiracial"
        case .other: return "Other"
        }
    }
}
