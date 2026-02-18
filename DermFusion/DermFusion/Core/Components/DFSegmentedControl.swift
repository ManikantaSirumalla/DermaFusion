//
//  DFSegmentedControl.swift
//  DermFusion
//
//  Styled segmented control wrapper using DermaFusion design tokens.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

struct DFSegmentedControl<Option: Hashable>: View {
    let title: String
    let options: [Option]
    let label: (Option) -> String
    @Binding var selection: Option

    var body: some View {
        Picker(title, selection: $selection) {
            ForEach(options, id: \.self) { option in
                Text(label(option)).tag(option)
            }
        }
        .pickerStyle(.segmented)
    }
}
