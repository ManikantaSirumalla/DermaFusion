//
//  AnalysisView.swift
//  DermFusion
//
//  Loading step between metadata and results. Centered icon with zoom + rotation.
//

import SwiftUI

/// Performs analysis and routes to results once complete.
struct AnalysisView: View {

    @ObservedObject var viewModel: ScanViewModel
    @EnvironmentObject private var appDataStore: AppDataStore
    @State private var navigateToResults = false
    @State private var isZoomingIn = true

    var body: some View {
        ZStack {
            DFDesignSystem.Colors.backgroundPrimary
                .ignoresSafeArea()

            VStack(spacing: 28) {
                appIconZoomView
                VStack(spacing: 6) {
                    Text("Working on your image…")
                        .font(.headline)
                    Text("Detecting skin features and patterns.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .navigationTitle("Analysis")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            guard viewModel.result == nil else { return }
            await viewModel.runAnalysis()
            navigateToResults = viewModel.result != nil
        }
        .background(
            NavigationLink(
                destination: ResultsView(
                    result: viewModel.result ?? .sample,
                    metadata: viewModel.metadataInput,
                    onSave: {
                        guard let result = viewModel.result, let metadata = viewModel.metadataInput else { return }
                        appDataStore.save(result: result, metadata: metadata)
                    },
                    onLearnMore: { appDataStore.selectedTab = .learn }
                ),
                isActive: $navigateToResults
            ) {
                EmptyView()
            }
            .hidden()
        )
        .hideTabBarWhenPushed()
    }

    private static let zoomCycleDuration: TimeInterval = 1.6
    private static let zoomOutPhaseDuration: TimeInterval = 0.8
    private static let rotationSpeedZoomOut: Double = 180
    private static let rotationSpeedZoomIn: Double = 72

    private static let iconBaseSize: CGFloat = 80
    private static let iconMaxScale: CGFloat = 1.15
    private static let iconContainerSize: CGFloat = iconBaseSize * iconMaxScale + 8

    @ViewBuilder
    private var appIconZoomView: some View {
        ZStack {
            TimelineView(.animation(minimumInterval: 1 / 30)) { timeline in
                let t = timeline.date.timeIntervalSinceReferenceDate
                let rotation = Self.rotationAtTime(t)
                iconImage
                    .scaleEffect(isZoomingIn ? Self.iconMaxScale : 0.85, anchor: .center)
                    .rotationEffect(.degrees(rotation), anchor: .center)
            }
        }
        .frame(width: Self.iconContainerSize, height: Self.iconContainerSize)
        .animation(
            .easeInOut(duration: 0.8).repeatForever(autoreverses: true),
            value: isZoomingIn
        )
        .onAppear { isZoomingIn = false }
    }

    private static func rotationAtTime(_ t: TimeInterval) -> Double {
        let cycleDuration = zoomCycleDuration
        let phaseDuration = zoomOutPhaseDuration
        let remainder = t.truncatingRemainder(dividingBy: cycleDuration)
        let fullCycles = (t / cycleDuration).rounded(.down)
        let rotationPerCycle = phaseDuration * rotationSpeedZoomOut
            + (cycleDuration - phaseDuration) * rotationSpeedZoomIn
        let rotationFromCycles = fullCycles * rotationPerCycle
        let rotationThisCycle: Double
        if remainder < phaseDuration {
            rotationThisCycle = remainder * rotationSpeedZoomOut
        } else {
            rotationThisCycle = phaseDuration * rotationSpeedZoomOut
                + (remainder - phaseDuration) * rotationSpeedZoomIn
        }
        return rotationFromCycles + rotationThisCycle
    }

    @ViewBuilder
    private var iconImage: some View {
        if UIImage(named: "icon") != nil {
            Image("icon")
                .resizable()
                .scaledToFit()
                .frame(width: 80, height: 80)
        } else {
            Image(systemName: "cross.case.fill")
                .font(.system(size: 56))
                .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
        }
    }
}

#Preview("Loading") {
    NavigationStack {
        AnalysisView(viewModel: ScanViewModel())
            .environmentObject(AppDataStore())
    }
}

#Preview("Dark") {
    NavigationStack {
        AnalysisView(viewModel: ScanViewModel())
            .environmentObject(AppDataStore())
    }
    .preferredColorScheme(.dark)
}
