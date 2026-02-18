//
//  View+ResponsiveLayout.swift
//  DermFusion
//
//  Responsive layout helpers to keep content consistent across iPhone and iPad sizes.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

extension View {
    /// Constrains content width for large screens while preserving full-width behavior on compact screens.
    func dfConstrainedContent(maxWidth: CGFloat = 760) -> some View {
        frame(maxWidth: maxWidth)
            .frame(maxWidth: .infinity, alignment: .center)
    }
}
