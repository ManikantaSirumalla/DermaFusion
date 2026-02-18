//
//  DFShadows.swift
//  DermFusion
//
//  Elevation definitions for cards and overlays.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Shadow token definitions.
enum DFShadows {
    static let card = ShadowStyle(color: Color.black.opacity(0.06), radius: 8, x: 0, y: 2)
    static let elevated = ShadowStyle(color: Color.black.opacity(0.10), radius: 16, x: 0, y: 4)
    static let modal = ShadowStyle(color: Color.black.opacity(0.16), radius: 24, x: 0, y: 8)
}

/// Reusable shadow style.
struct ShadowStyle {
    let color: Color
    let radius: CGFloat
    let x: CGFloat
    let y: CGFloat
}

extension View {
    /// Applies a design-system shadow token.
    func dfShadow(_ style: ShadowStyle) -> some View {
        shadow(color: style.color, radius: style.radius, x: style.x, y: style.y)
    }
}
