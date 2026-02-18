//
//  DFSlider.swift
//  DermFusion
//
//  Styled slider with numeric value readout.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

struct DFSlider: View {
    let title: String
    @Binding var value: Double
    let range: ClosedRange<Double>

    var body: some View {
        VStack(alignment: .leading, spacing: DFDesignSystem.Spacing.xs) {
            HStack {
                Text(title)
                Spacer()
                Text("\(Int(value))")
                    .font(DFDesignSystem.Typography.mono)
            }
            Slider(value: $value, in: range, step: 1)
                .tint(DFDesignSystem.Colors.brandPrimary)
        }
    }
}
