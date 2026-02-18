//
//  GradCAMOverlayView.swift
//  DermFusion
//
//  Overlay container for original image and GradCAM heatmap.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Shows preprocessed image by default; toggles to GradCAM overlay when showGradCAM is true.
struct GradCAMOverlayView: View {
    @Binding var showGradCAM: Bool

    var body: some View {
        Image(showGradCAM ? "gradcam_overlay" : "preprocessed")
            .resizable()
            .aspectRatio(contentMode: .fit)
            .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadius, style: .continuous))
    }
}
