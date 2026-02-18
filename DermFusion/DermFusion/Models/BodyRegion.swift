//
//  BodyRegion.swift
//  DermFusion
//
//  Body region options from the design document's MVP metadata flow.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

/// Anatomical regions used in lesion-location metadata collection.
enum BodyRegion: String, CaseIterable, Identifiable, Sendable {
    case scalp
    case face
    case ear
    case neck
    case chest
    case abdomen
    case back
    case upperExtremity
    case lowerExtremity
    case hand
    case foot
    case genital

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .scalp:
            return "Scalp"
        case .face:
            return "Face"
        case .ear:
            return "Ear"
        case .neck:
            return "Neck"
        case .chest:
            return "Chest"
        case .abdomen:
            return "Abdomen"
        case .back:
            return "Back"
        case .upperExtremity:
            return "Upper Extremity"
        case .lowerExtremity:
            return "Lower Extremity"
        case .hand:
            return "Hand"
        case .foot:
            return "Foot"
        case .genital:
            return "Genital"
        }
    }

    /// Scene node name expected in `BodyModel.scn`.
    var sceneNodeName: String {
        switch self {
        case .scalp:
            return "region_scalp"
        case .face:
            return "region_face"
        case .ear:
            return "region_ear"
        case .neck:
            return "region_neck"
        case .chest:
            return "region_chest"
        case .abdomen:
            return "region_abdomen"
        case .back:
            return "region_back"
        case .upperExtremity:
            return "region_upper_extremity"
        case .lowerExtremity:
            return "region_lower_extremity"
        case .hand:
            return "region_hand"
        case .foot:
            return "region_foot"
        case .genital:
            return "region_genital"
        }
    }

    /// Initializes a body region from a SceneKit node name.
    init?(sceneNodeName: String) {
        switch sceneNodeName {
        case "region_scalp":
            self = .scalp
        case "region_face":
            self = .face
        case "region_ear":
            self = .ear
        case "region_neck":
            self = .neck
        case "region_chest":
            self = .chest
        case "region_abdomen":
            self = .abdomen
        case "region_back":
            self = .back
        case "region_upper_extremity":
            self = .upperExtremity
        case "region_lower_extremity":
            self = .lowerExtremity
        case "region_hand":
            self = .hand
        case "region_foot":
            self = .foot
        case "region_genital":
            self = .genital
        default:
            if let matchedRegion = BodyRegion.fuzzyMatch(from: sceneNodeName) {
                self = matchedRegion
            } else {
                return nil
            }
        }
    }

    /// Attempts to infer a body region from non-standard node naming patterns.
    private static func fuzzyMatch(from sceneNodeName: String) -> BodyRegion? {
        let normalized = sceneNodeName
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: "-", with: "_")
            .replacingOccurrences(of: " ", with: "_")

        if normalized.contains("scalp") || normalized.contains("head_top") || normalized == "head" {
            return .scalp
        }
        if normalized.contains("face") || normalized.contains("forehead") || normalized.contains("nose") || normalized.contains("cheek") {
            return .face
        }
        if normalized.contains("ear") {
            return .ear
        }
        if normalized.contains("neck") {
            return .neck
        }
        if normalized.contains("chest") || normalized.contains("thorax") || normalized.contains("torso_front") || normalized.contains("pectoral") {
            return .chest
        }
        if normalized.contains("abdomen") || normalized.contains("belly") || normalized.contains("stomach") {
            return .abdomen
        }
        if normalized == "back" || normalized.contains("upper_back") || normalized.contains("lower_back") || normalized.contains("spine") || normalized.contains("torso_back") {
            return .back
        }
        if normalized.contains("upper_extremity") || normalized.contains("arm") || normalized.contains("shoulder") || normalized.contains("forearm") {
            return .upperExtremity
        }
        if normalized.contains("hand") || normalized.contains("wrist") || normalized.contains("palm") || normalized.contains("finger") {
            return .hand
        }
        if normalized.contains("lower_extremity") || normalized.contains("thigh") || normalized.contains("knee") || normalized.contains("leg") || normalized.contains("calf") {
            return .lowerExtremity
        }
        if normalized.contains("foot") || normalized.contains("ankle") || normalized.contains("toe") {
            return .foot
        }
        if normalized.contains("genital") || normalized.contains("groin") || normalized.contains("pubic") {
            return .genital
        }
        return nil
    }
}
