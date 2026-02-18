//
//  MetadataInput.swift
//  DermFusion
//
//  Typed metadata payload collected from users before model inference.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

/// Metadata entered by the user for a single scan.
struct MetadataInput: Sendable {

    // MARK: - Properties

    let age: Int?
    let sex: Sex
    let bodyRegion: BodyRegion
    let name: String?
    let ethnicity: String?

    init(age: Int?, sex: Sex, bodyRegion: BodyRegion, name: String? = nil, ethnicity: String? = nil) {
        self.age = age
        self.sex = sex
        self.bodyRegion = bodyRegion
        self.name = name
        self.ethnicity = ethnicity
    }
}
