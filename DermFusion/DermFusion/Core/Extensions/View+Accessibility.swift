//
//  View+Accessibility.swift
//  DermFusion
//
//  Reusable accessibility helpers for view semantics.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

extension View {
    func dfAccessibleGroup() -> some View {
        accessibilityElement(children: .combine)
    }
}
