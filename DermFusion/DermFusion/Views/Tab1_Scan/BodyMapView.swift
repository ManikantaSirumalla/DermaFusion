//
//  BodyMapView.swift
//  DermFusion
//
//  Body-region selection screen: 3D body mesh with tap-to-select and Reset.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Interactive body map for lesion location selection.
struct BodyMapView: View {

    // MARK: - Dependencies

    let sex: Sex
    @Binding var selectedRegion: BodyRegion?
    @Environment(\.dismiss) private var dismiss

    // MARK: - Local State

    @StateObject private var viewModel = BodyMapViewModel()
    @State private var selectedRegionDisplayName: String?
    @State private var hasBodyModel = false
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize

    // MARK: - Body

    var body: some View {
        ZStack {
            Color(uiColor: .systemBackground)
                .ignoresSafeArea()

            if hasBodyModel {
                bodyMapContent
            } else {
                fallbackRegionSelector
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle("Select Location")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            viewModel.selectedRegion = selectedRegion
            hasBodyModel = BodyMeshViewerRepresentable.hasMesh(for: sex)
        }
        .onChange(of: viewModel.selectedRegion) { _, newValue in
            selectedRegion = newValue
        }
        .onChange(of: sex) { _, newSex in
            hasBodyModel = BodyMeshViewerRepresentable.hasMesh(for: newSex)
        }
        .hideTabBarWhenPushed()
    }

    private var bodyMapContent: some View {
        GeometryReader { geometry in
            VStack(spacing: 0) {
                Text("Tap on the body to select the affected area")
                    .font(.system(size: 13))
                    .foregroundStyle(Color.secondary)

                if horizontalSizeClass == .regular {
                    Spacer()
                        .frame(minHeight: topBottomMargin(in: geometry))
                } else {
                    Spacer(minLength: 0)
                }

                BodyMeshViewerRepresentable(sex: sex) { tapResult in
                    viewModel.selectedRegion = tapResult.mappedBodyRegion
                    selectedRegionDisplayName = tapResult.regionDisplayName
                }
                .frame(maxWidth: .infinity)
                .frame(height: viewerHeight(in: geometry))
                .clipShape(RoundedRectangle(cornerRadius: 16))
                .padding(.horizontal, contentHorizontalPadding)

                if horizontalSizeClass == .regular {
                    Spacer()
                        .frame(minHeight: topBottomMargin(in: geometry))
                } else {
                    Spacer(minLength: 0)
                }

            VStack(spacing: 10) {
                if let name = selectedRegionDisplayName {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(Color.cyan)
                            .frame(width: 8, height: 8)
                        Text(name)
                            .font(.system(size: 15, weight: .medium))
                            .foregroundStyle(Color.primary)
                    }
                    .padding(.top, 8)
                } else {
                    Text("No area selected")
                        .font(.system(size: 14))
                        .foregroundStyle(Color.secondary)
                        .padding(.top, 8)
                }

                Button {
                    viewModel.selectedRegion = nil
                    selectedRegionDisplayName = nil
                    NotificationCenter.default.post(name: .resetBodyView, object: nil)
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: "arrow.counterclockwise")
                            .font(.system(size: 12, weight: .medium))
                        Text("Reset View")
                            .font(.system(size: 13, weight: .medium))
                    }
                    .foregroundStyle(Color.secondary)
                }
                .buttonStyle(.plain)
                .padding(.top, 2)

                Button {
                    dismiss()
                } label: {
                    Text(viewModel.selectedRegion != nil ? "Confirm: \(viewModel.selectedRegion!.displayName)" : "Select a spot to continue")
                        .font(.system(size: 17, weight: .semibold))
                        .foregroundStyle(.white)
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 14)
                        .background(viewModel.selectedRegion != nil ? Color.blue : Color.gray.opacity(0.3))
                        .cornerRadius(12)
                }
                .buttonStyle(.plain)
                .disabled(viewModel.selectedRegion == nil)
                .padding(.horizontal, contentHorizontalPadding)
                .padding(.bottom, 8)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .dfConstrainedContent(maxWidth: contentMaxWidth(in: geometry))
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func contentMaxWidth(in geometry: GeometryProxy) -> CGFloat {
        if horizontalSizeClass == .regular {
            return geometry.size.width * 0.95
        }
        return 430
    }

    private var contentHorizontalPadding: CGFloat {
        if dynamicTypeSize.isAccessibilitySize {
            return DFDesignSystem.Spacing.md
        }
        return horizontalSizeClass == .regular ? DFDesignSystem.Spacing.lg : DFDesignSystem.Spacing.screenHorizontal
    }

    /// Space reserved for hint (top) and label + Reset + Confirm (bottom) so 3%–94%–3% fits.
    private var topReservedHeight: CGFloat { 28 }
    private var bottomSectionReservedHeight: CGFloat {
        horizontalSizeClass == .regular ? 220 : 200
    }

    /// Content height for the 3%–94%–3% area (available minus hint and bottom section).
    private func contentHeight(in geometry: GeometryProxy) -> CGFloat {
        let reserved = topReservedHeight + bottomSectionReservedHeight
        return max(400, geometry.size.height - reserved)
    }

    /// Margin at top and bottom on iPad (3% each so mesh can be 94%).
    private func topBottomMargin(in geometry: GeometryProxy) -> CGFloat {
        guard horizontalSizeClass == .regular else { return 0 }
        return contentHeight(in: geometry) * 0.03
    }

    /// Viewer height: iPhone fixed; iPad = 94% of content height (3% top + 94% mesh + 3% bottom).
    private func viewerHeight(in geometry: GeometryProxy) -> CGFloat {
        let available = geometry.size.height
        if horizontalSizeClass == .regular {
            return contentHeight(in: geometry) * 0.94
        }
        return dynamicTypeSize.isAccessibilitySize ? 520 : 620
    }

    private var fallbackRegionSelector: some View {
        ScrollView {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 160), spacing: DFDesignSystem.Spacing.sm)], spacing: DFDesignSystem.Spacing.sm) {
                ForEach(BodyRegion.allCases, id: \.id) { region in
                    Button {
                        viewModel.selectedRegion = region
                        selectedRegionDisplayName = region.displayName
                    } label: {
                        Text(region.displayName)
                            .font(DFDesignSystem.Typography.bodyBold)
                            .frame(maxWidth: .infinity, minHeight: DFDesignSystem.Spacing.touchTarget)
                            .foregroundStyle(viewModel.selectedRegion == region ? DFDesignSystem.Colors.textInverse : DFDesignSystem.Colors.textPrimary)
                            .background(viewModel.selectedRegion == region ? DFDesignSystem.Colors.brandPrimary : DFDesignSystem.Colors.backgroundSecondary)
                            .clipShape(RoundedRectangle(cornerRadius: DFDesignSystem.Spacing.cornerRadiusSmall))
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Select \(region.displayName)")
                }
            }
            .padding(.vertical, DFDesignSystem.Spacing.xs)
        }
        .padding(.horizontal, contentHorizontalPadding)
    }
}

#Preview {
    NavigationStack {
        BodyMapView(sex: .male, selectedRegion: .constant(nil))
    }
}
