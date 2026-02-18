//
//  DFTypography.swift
//  DermFusion
//
//  Semantic typography tokens with Dynamic Type support.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Typography scale for DermaFusion.
enum DFTypography {
    static let displayLarge = Font.system(size: 34, weight: .bold, design: .rounded)
    static let displayMedium = Font.system(size: 28, weight: .semibold, design: .default)
    static let headline = Font.title2.weight(.semibold)
    static let title2 = Font.title2
    static let subheadline = Font.headline
    static let bodyRegular = Font.body
    static let body = Font.body
    static let bodyBold = Font.body.weight(.semibold)
    static let caption = Font.caption
    static let captionSmall = Font.caption2
    static let disclaimer = Font.footnote
    static let mono = Font.system(size: 14, weight: .medium, design: .monospaced)
}
