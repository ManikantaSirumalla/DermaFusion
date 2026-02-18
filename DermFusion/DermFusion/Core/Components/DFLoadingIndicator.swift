//
//  DFLoadingIndicator.swift
//  DermFusion
//
//  Branded loading indicator used during analysis and async workflows.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

struct DFLoadingIndicator: View {
    var body: some View {
        ProgressView()
            .tint(DFDesignSystem.Colors.brandPrimary)
            .scaleEffect(1.2)
            .frame(minWidth: DFDesignSystem.Spacing.touchTarget, minHeight: DFDesignSystem.Spacing.touchTarget)
    }
}
