//
//  ScanHomeView.swift
//  DermFusion
//
//  Scan-tab landing: New Scan and Choose from Library trigger camera or picker directly.
//

import AVFoundation
import Photos
import PhotosUI
import SwiftUI
import UIKit

/// Landing screen for the Scan tab.
struct ScanHomeView: View {

    @EnvironmentObject private var appDataStore: AppDataStore
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass
    @Environment(\.dynamicTypeSize) private var dynamicTypeSize
    @Environment(\.openURL) private var openURL

    @State private var showLiveCamera = false
    @State private var capturedImage: UIImage?
    @State private var navigateToImageReview = false
    @State private var showPermissionAlert = false
    @State private var alertTitle = ""
    @State private var alertMessage = ""
    @State private var selectedPhotoItems: [PhotosPickerItem] = []

    var body: some View {
        List {
            Section {
                heroView
                    .listRowBackground(EmptyView())
                    .listRowSeparator(.hidden)
                    .listRowInsets(EdgeInsets(top: 8, leading: 16, bottom: 8, trailing: 16))
            }

            Section {
                Button {
                    Task { await handleCameraTapped() }
                } label: {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("New Scan")
                                .font(.body)
                                .fontWeight(.medium)
                            Text("Capture with camera and review before analysis.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: DFDesignSystem.Icons.capture)
                            .font(.body)
                            .foregroundStyle(.white)
                            .frame(width: 36, height: 36)
                            .background(DFDesignSystem.Colors.brandPrimary)
                            .clipShape(Circle())
                    }
                }
                .buttonStyle(.plain)

                PhotosPicker(
                    selection: $selectedPhotoItems,
                    maxSelectionCount: 1,
                    matching: .images
                ) {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Choose from Library")
                                .font(.body)
                                .fontWeight(.medium)
                            Text("Import an existing lesion image for review.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: DFDesignSystem.Icons.gallery)
                            .font(.body)
                            .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                            .frame(width: 36, height: 36)
                            .background(DFDesignSystem.Colors.backgroundSecondary)
                            .clipShape(Circle())
                    }
                }
            } header: {
                Text("Start")
            } footer: {
                Text("⚠️ For education and research only. Not a substitute for professional diagnosis — always consult a dermatologist for skin concerns.")
                    .font(.caption)
                    .foregroundStyle(Color.yellow)
            }

            if let record = appDataStore.latestRecord {
                Section {
                    Button {
                        appDataStore.selectedTab = .history
                    } label: {
                        HStack(spacing: 12) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text("Last Scan")
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                Text("\(record.primaryDiagnosis) — \(Int((record.predictions.values.max() ?? 0) * 100))%")
                                    .font(.body)
                                    .fontWeight(.medium)
                                Text(record.timestamp.relativeDescription)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            DFRiskBadge(level: riskLevel(from: record.riskLevel), compact: true)
                        }
                        .padding(.vertical, 4)
                    }
                    .buttonStyle(.plain)
                } header: {
                    Text("Recent")
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Scan")
        .navigationBarTitleDisplayMode(.large)
        .showTabBarWhenRoot()
        .background(
            NavigationLink(
                destination: ImageReviewView(
                    image: capturedImage,
                    onRetake: {
                        navigateToImageReview = false
                        Task { @MainActor in
                            try? await Task.sleep(nanoseconds: 180_000_000)
                            showLiveCamera = true
                        }
                    }
                ),
                isActive: $navigateToImageReview
            ) {
                EmptyView()
            }
            .hidden()
        )
        .fullScreenCover(isPresented: $showLiveCamera) {
            LiveCameraCaptureView(
                onCancel: { showLiveCamera = false },
                onPhotoCaptured: { image in
                    capturedImage = image
                    showLiveCamera = false
                    navigateToImageReview = true
                }
            )
        }
        .onChange(of: selectedPhotoItems) { _, newItems in
            guard let item = newItems.first else { return }
            Task {
                await loadImage(from: item)
            }
        }
        .alert(alertTitle, isPresented: $showPermissionAlert) {
            Button("Not Now", role: .cancel) {}
            Button("Open Settings") { openAppSettings() }
        } message: {
            Text(alertMessage)
        }
        .onChange(of: appDataStore.shouldDismissScanFlow) { _, shouldDismiss in
            if shouldDismiss {
                navigateToImageReview = false
                appDataStore.shouldDismissScanFlow = false
            }
        }
    }

    @MainActor
    private func handleCameraTapped() async {
        let status = AVCaptureDevice.authorizationStatus(for: .video)
        switch status {
        case .authorized:
            showLiveCamera = true
        case .notDetermined:
            let granted = await requestCameraAccess()
            if granted { showLiveCamera = true }
            else {
                presentPermissionAlert(
                    title: "Camera Access Needed",
                    message: "DermaFusion needs camera access to capture lesion images. You can enable this in Settings."
                )
            }
        case .denied, .restricted:
            presentPermissionAlert(
                title: "Camera Access Needed",
                message: "Camera access is currently disabled. Enable it in Settings to continue."
            )
        @unknown default:
            presentPermissionAlert(
                title: "Camera Unavailable",
                message: "Camera access could not be determined. Please try again."
            )
        }
    }

    private func loadImage(from item: PhotosPickerItem) async {
        guard let data = try? await item.loadTransferable(type: Data.self),
              let image = UIImage(data: data) else { return }
        await MainActor.run {
            capturedImage = image
            selectedPhotoItems = []
            navigateToImageReview = true
        }
    }

    private func requestCameraAccess() async -> Bool {
        await withCheckedContinuation { continuation in
            AVCaptureDevice.requestAccess(for: .video) { continuation.resume(returning: $0) }
        }
    }

    @MainActor
    private func presentPermissionAlert(title: String, message: String) {
        alertTitle = title
        alertMessage = message
        showPermissionAlert = true
    }

    private func openAppSettings() {
        guard let url = URL(string: UIApplication.openSettingsURLString) else { return }
        openURL(url)
    }

    private var heroView: some View {
        VStack(spacing: 12) {
            appIcon
            Text("DermaFusion")
                .font(.title)
                .fontWeight(.bold)
            Text("On-device multi-modal skin lesion classification")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 20)
    }

    private var appIcon: some View {
        Group {
            if UIImage(named: "icon") != nil {
                Image("icon")
                    .resizable()
                    .scaledToFit()
                    .frame(width: appIconSize, height: appIconSize)
            } else {
                Image(systemName: "cross.case.fill")
                    .font(.largeTitle)
                    .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
            }
        }
    }

    private func riskLevel(from raw: String) -> RiskLevel {
        RiskLevel(rawValue: raw) ?? .moderate
    }

    private var appIconSize: CGFloat {
        dynamicTypeSize.isAccessibilitySize ? 96 : (horizontalSizeClass == .regular ? 140 : 120)
    }
}

#Preview("With Last Scan") {
    let store = AppDataStore()
    store.save(result: .sample, metadata: MetadataInput(age: 55, sex: .male, bodyRegion: .back))
    return NavigationStack { ScanHomeView() }
        .environmentObject(store)
}

#Preview("Dark") {
    NavigationStack { ScanHomeView() }
        .environmentObject(AppDataStore())
        .preferredColorScheme(.dark)
}
