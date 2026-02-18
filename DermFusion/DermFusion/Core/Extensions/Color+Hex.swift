//
//  Color+Hex.swift
//  DermFusion
//
//  Hex string initializer for SwiftUI Color values.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

extension Color {
    init?(hex: String) {
        let hex = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        guard hex.count == 6, let int = Int(hex, radix: 16) else { return nil }
        let red = Double((int >> 16) & 0xFF) / 255.0
        let green = Double((int >> 8) & 0xFF) / 255.0
        let blue = Double(int & 0xFF) / 255.0
        self.init(red: red, green: green, blue: blue)
    }
}
