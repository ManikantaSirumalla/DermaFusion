//
//  LiveCameraCaptureView.swift
//  DermFusion
//
//  Full-screen AVFoundation camera capture flow for lesion imaging.
//
//  Created by Manikanta Sirumalla on 2/14/26.
//

import AVFoundation
import Combine
import SwiftUI
import UIKit

/// Full-screen camera capture view with framing guide and capture controls.
struct LiveCameraCaptureView: View {

    // MARK: - Dependencies

    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var cameraController = CameraSessionController()

    // MARK: - Inputs

    let onCancel: () -> Void
    let onPhotoCaptured: (UIImage) -> Void

    // MARK: - Local State

    @State private var guidePulse = false
    @State private var showGuidanceOverlay = true
    @State private var error: DFError?

    // MARK: - Body

    var body: some View {
        GeometryReader { geometry in
            let minDimension = min(geometry.size.width, geometry.size.height)
            let guideDiameter = min(max(220, minDimension * 0.62), 480)
            let guideRect = CGRect(
                x: (geometry.size.width - guideDiameter) / 2,
                y: (geometry.size.height - guideDiameter) / 2,
                width: guideDiameter,
                height: guideDiameter
            )

            ZStack {
                CameraPreviewView(session: cameraController.session)
                    .ignoresSafeArea()

                FramingOverlayView(
                    diameter: guideDiameter,
                    pulse: guidePulse
                )
                .allowsHitTesting(false)

                guidanceOverlay

                VStack {
                    topControls
                    Spacer()
                    Text("Position the lesion within the circle")
                        .font(DFDesignSystem.Typography.subheadline)
                        .foregroundStyle(DFDesignSystem.Colors.textInverse)
                        .padding(.horizontal, DFDesignSystem.Spacing.md)
                        .padding(.vertical, DFDesignSystem.Spacing.xs)
                        .background(.ultraThinMaterial)
                        .clipShape(Capsule())
                    captureButton
                        .padding(.top, DFDesignSystem.Spacing.md)
                        .padding(.bottom, DFDesignSystem.Spacing.xxl)
                }
                .padding(.horizontal, DFDesignSystem.Spacing.screenHorizontal)
                .padding(.top, geometry.safeAreaInsets.top + DFDesignSystem.Spacing.md)
                .padding(.bottom, geometry.safeAreaInsets.bottom + DFDesignSystem.Spacing.sm)
                .frame(maxWidth: 960)
                .frame(maxWidth: .infinity)
            }
            .background(Color.black)
            .ignoresSafeArea()
            .task {
                bindCallbacks(previewSize: geometry.size, guideRect: guideRect)
                cameraController.configureSessionIfNeeded()
                cameraController.startSession()
                withAnimation(.easeInOut(duration: 1.25).repeatForever(autoreverses: true)) {
                    guidePulse = true
                }
            }
            .onAppear {
                DispatchQueue.main.asyncAfter(deadline: .now() + 5) {
                    withAnimation(.easeOut(duration: 0.3)) {
                        showGuidanceOverlay = false
                    }
                }
            }
            .onDisappear {
                cameraController.stopSession()
            }
            .onChange(of: scenePhase) { _, newValue in
                switch newValue {
                case .active:
                    cameraController.startSession()
                case .inactive, .background:
                    cameraController.stopSession()
                @unknown default:
                    break
                }
            }
            .alert(item: $error) { item in
                Alert(
                    title: Text("Capture Error"),
                    message: Text(item.localizedDescription),
                    dismissButton: .default(Text("OK"))
                )
            }
        }
    }

    // MARK: - Subviews

    private var guidanceOverlay: some View {
        VStack {
            Group {
                if showGuidanceOverlay {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "lightbulb.fill")
                            .font(DFDesignSystem.Typography.body)
                            .foregroundStyle(.yellow)
                            .symbolRenderingMode(.hierarchical)
                        Text("Use clear lighting and center the lesion for the most consistent classification result.")
                            .font(DFDesignSystem.Typography.caption)
                            .foregroundStyle(DFDesignSystem.Colors.textInverse)
                    }
                    .padding(DFDesignSystem.Spacing.md)
                    .frame(maxWidth: .infinity)
                    .background(.ultraThinMaterial)
                    .transition(.opacity.combined(with: .move(edge: .top)))
                }
            }
            .animation(.easeOut(duration: 0.3), value: showGuidanceOverlay)
            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .allowsHitTesting(false)
    }

    private var topControls: some View {
        HStack {
            Button("Cancel") {
                onCancel()
            }
            .font(DFDesignSystem.Typography.bodyBold)
            .foregroundStyle(DFDesignSystem.Colors.textInverse)
            .padding(.horizontal, DFDesignSystem.Spacing.sm)
            .padding(.vertical, DFDesignSystem.Spacing.xs)
            .background(Color.black.opacity(0.35))
            .clipShape(Capsule())

            Spacer()

            HStack(spacing: DFDesignSystem.Spacing.sm) {
                Button {
                    cameraController.toggleFlash()
                } label: {
                    Image(systemName: cameraController.isFlashEnabled ? "bolt.fill" : "bolt.slash.fill")
                        .font(DFDesignSystem.Typography.headline)
                        .foregroundStyle(DFDesignSystem.Colors.textInverse)
                        .frame(width: DFDesignSystem.Spacing.touchTarget, height: DFDesignSystem.Spacing.touchTarget)
                        .background(Color.black.opacity(0.35))
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel(cameraController.isFlashEnabled ? "Disable flash" : "Enable flash")

                Button {
                    cameraController.toggleCameraPosition()
                } label: {
                    Image(systemName: "arrow.triangle.2.circlepath.camera")
                        .font(DFDesignSystem.Typography.headline)
                        .foregroundStyle(DFDesignSystem.Colors.textInverse)
                        .frame(width: DFDesignSystem.Spacing.touchTarget, height: DFDesignSystem.Spacing.touchTarget)
                        .background(Color.black.opacity(0.35))
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Switch camera")
            }
        }
    }

    private var captureButton: some View {
        Button {
            cameraController.capturePhoto()
        } label: {
            ZStack {
                Circle()
                    .stroke(Color.black.opacity(0.28), lineWidth: 2)
                    .frame(width: 72, height: 72)
                Circle()
                    .fill(Color.white)
                    .frame(width: 62, height: 62)
            }
            .shadow(color: .black.opacity(0.22), radius: 8, x: 0, y: 3)
        }
        .buttonStyle(.plain)
        .disabled(cameraController.isCaptureInProgress)
        .opacity(cameraController.isCaptureInProgress ? 0.6 : 1.0)
        .accessibilityLabel("Capture image")
    }

    // MARK: - Private Helpers

    private func bindCallbacks(previewSize: CGSize, guideRect: CGRect) {
        cameraController.onPhotoCaptured = { image in
            DispatchQueue.global(qos: .userInitiated).async {
                let guidedCrop = GuidedImageCropper.cropToGuideCircle(
                    image: image,
                    previewSize: previewSize,
                    guideRect: guideRect
                )
                DispatchQueue.main.async {
                    onPhotoCaptured(guidedCrop)
                }
            }
        }
        cameraController.onError = { cameraError in
            error = cameraError
        }
    }
}

/// Overlay mask with a circular transparent guide.
private struct FramingOverlayView: View {
    let diameter: CGFloat
    let pulse: Bool

    var body: some View {
        GeometryReader { proxy in
            let rect = proxy.frame(in: .local)
            let circleRect = CGRect(
                x: (rect.width - diameter) / 2,
                y: (rect.height - diameter) / 2,
                width: diameter,
                height: diameter
            )

            ZStack {
                FramingMaskShape(circleRect: circleRect)
                    .fill(Color.black.opacity(0.52), style: FillStyle(eoFill: true))

                Circle()
                    .stroke(Color.white.opacity(0.85), lineWidth: 2)
                    .frame(width: diameter, height: diameter)
                    .scaleEffect(pulse ? 1.03 : 0.97)
            }
        }
    }
}

private struct FramingMaskShape: Shape {
    let circleRect: CGRect

    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.addRect(rect)
        path.addEllipse(in: circleRect)
        return path
    }
}

/// UIKit wrapper that renders an AVCaptureVideoPreviewLayer.
private struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession

    func makeUIView(context: Context) -> PreviewContainerView {
        let view = PreviewContainerView()
        view.previewLayer.videoGravity = .resizeAspectFill
        view.previewLayer.session = session
        return view
    }

    func updateUIView(_ uiView: PreviewContainerView, context: Context) {
        uiView.previewLayer.session = session
    }
}

private final class PreviewContainerView: UIView {
    override class var layerClass: AnyClass {
        AVCaptureVideoPreviewLayer.self
    }

    var previewLayer: AVCaptureVideoPreviewLayer {
        guard let layer = layer as? AVCaptureVideoPreviewLayer else {
            fatalError("Expected AVCaptureVideoPreviewLayer")
        }
        return layer
    }
}

@MainActor
private final class CameraSessionController: NSObject, ObservableObject {

    // MARK: - Public State

    @Published private(set) var isCaptureInProgress = false
    @Published private(set) var isFlashEnabled = false

    let session = AVCaptureSession()
    var onPhotoCaptured: ((UIImage) -> Void)?
    var onError: ((DFError) -> Void)?

    // MARK: - Private State

    private let sessionQueue = DispatchQueue(label: "com.dermfusion.camera.session")
    private let photoOutput = AVCapturePhotoOutput()
    private var currentInput: AVCaptureDeviceInput?
    private var isConfigured = false
    private var usingFrontCamera = false

    // MARK: - Session Lifecycle

    func configureSessionIfNeeded() {
        guard !isConfigured else { return }
        sessionQueue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .photo

            guard let device = self.cameraDevice(position: .back) else {
                DispatchQueue.main.async {
                    self.onError?(
                        .generic(
                            title: "Camera Unavailable",
                            message: "No camera device was found on this device."
                        )
                    )
                }
                self.session.commitConfiguration()
                return
            }

            do {
                let input = try AVCaptureDeviceInput(device: device)
                if self.session.canAddInput(input) {
                    self.session.addInput(input)
                    self.currentInput = input
                }
            } catch {
                DispatchQueue.main.async {
                    self.onError?(
                        .generic(
                            title: "Camera Setup Failed",
                            message: "Unable to initialize camera input."
                        )
                    )
                }
            }

            if self.session.canAddOutput(self.photoOutput) {
                self.session.addOutput(self.photoOutput)
            }
            self.photoOutput.isHighResolutionCaptureEnabled = true

            self.session.commitConfiguration()
            self.isConfigured = true
        }
    }

    func startSession() {
        sessionQueue.async { [weak self] in
            guard let self, self.isConfigured, !self.session.isRunning else { return }
            self.session.startRunning()
        }
    }

    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self, self.session.isRunning else { return }
            self.session.stopRunning()
        }
    }

    // MARK: - Actions

    func toggleFlash() {
        isFlashEnabled.toggle()
    }

    func toggleCameraPosition() {
        sessionQueue.async { [weak self] in
            guard let self, self.isConfigured, let currentInput = self.currentInput else { return }

            let nextPosition: AVCaptureDevice.Position = self.usingFrontCamera ? .back : .front
            guard let nextDevice = self.cameraDevice(position: nextPosition) else { return }

            do {
                let nextInput = try AVCaptureDeviceInput(device: nextDevice)
                self.session.beginConfiguration()
                self.session.removeInput(currentInput)
                if self.session.canAddInput(nextInput) {
                    self.session.addInput(nextInput)
                    self.currentInput = nextInput
                    self.usingFrontCamera.toggle()
                } else {
                    self.session.addInput(currentInput)
                }
                self.session.commitConfiguration()
            } catch {
                DispatchQueue.main.async {
                    self.onError?(
                        .generic(
                            title: "Camera Switch Failed",
                            message: "Could not switch camera. Please try again."
                        )
                    )
                }
            }
        }
    }

    func capturePhoto() {
        guard !isCaptureInProgress else { return }
        isCaptureInProgress = true

        sessionQueue.async { [weak self] in
            guard let self else { return }
            let settings = AVCapturePhotoSettings()
            settings.flashMode = self.isFlashEnabled ? .on : .off
            settings.isHighResolutionPhotoEnabled = true
            self.photoOutput.capturePhoto(with: settings, delegate: self)
        }
    }

    // MARK: - Helpers

    private func cameraDevice(position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position)
    }
}

extension CameraSessionController: AVCapturePhotoCaptureDelegate {
    nonisolated func photoOutput(
        _ output: AVCapturePhotoOutput,
        didFinishProcessingPhoto photo: AVCapturePhoto,
        error: Error?
    ) {
        if error != nil {
            Task { @MainActor in
                self.isCaptureInProgress = false
                self.onError?(
                    .generic(
                        title: "Capture Failed",
                        message: "Unable to process the captured image."
                    )
                )
            }
            return
        }

        guard let data = photo.fileDataRepresentation(), let image = UIImage(data: data) else {
            Task { @MainActor in
                self.isCaptureInProgress = false
                self.onError?(
                    .generic(
                        title: "Capture Failed",
                        message: "The captured image could not be read."
                    )
                )
            }
            return
        }

        Task { @MainActor in
            self.isCaptureInProgress = false
            self.onPhotoCaptured?(image)
        }
    }
}

#Preview {
    LiveCameraCaptureView(
        onCancel: {},
        onPhotoCaptured: { _ in }
    )
}

private enum GuidedImageCropper {
    static func cropToGuideCircle(image: UIImage, previewSize: CGSize, guideRect: CGRect) -> UIImage {
        let normalized = normalizeIfNeeded(image)
        guard let cgImage = normalized.cgImage else { return normalized }

        let rawSize = CGSize(width: cgImage.width, height: cgImage.height)
        let scale = max(previewSize.width / rawSize.width, previewSize.height / rawSize.height)
        let displayedSize = CGSize(width: rawSize.width * scale, height: rawSize.height * scale)
        let displayedOrigin = CGPoint(
            x: (previewSize.width - displayedSize.width) / 2,
            y: (previewSize.height - displayedSize.height) / 2
        )

        var cropRect = CGRect(
            x: (guideRect.origin.x - displayedOrigin.x) / scale,
            y: (guideRect.origin.y - displayedOrigin.y) / scale,
            width: guideRect.width / scale,
            height: guideRect.height / scale
        )

        let imageBounds = CGRect(origin: .zero, size: rawSize)
        cropRect = cropRect.intersection(imageBounds).integral

        guard
            cropRect.width > 1,
            cropRect.height > 1,
            let croppedCgImage = cgImage.cropping(to: cropRect)
        else {
            return normalized
        }

        let squareImage = UIImage(cgImage: croppedCgImage, scale: normalized.scale, orientation: .up)
        return circularMask(image: squareImage)
    }

    private static func circularMask(image: UIImage) -> UIImage {
        let side = min(image.size.width, image.size.height)
        let cropOrigin = CGPoint(
            x: (image.size.width - side) / 2,
            y: (image.size.height - side) / 2
        )
        let cropRect = CGRect(origin: cropOrigin, size: CGSize(width: side, height: side))

        let format = UIGraphicsImageRendererFormat.default()
        format.scale = image.scale
        format.opaque = false

        let renderer = UIGraphicsImageRenderer(size: CGSize(width: side, height: side), format: format)
        return renderer.image { _ in
            let clipPath = UIBezierPath(ovalIn: CGRect(x: 0, y: 0, width: side, height: side))
            clipPath.addClip()
            image.draw(
                in: CGRect(
                    x: -cropRect.origin.x,
                    y: -cropRect.origin.y,
                    width: image.size.width,
                    height: image.size.height
                )
            )
        }
    }

    private static func normalizeIfNeeded(_ image: UIImage) -> UIImage {
        guard image.imageOrientation != .up else { return image }
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = image.scale
        return UIGraphicsImageRenderer(size: image.size, format: format).image { _ in
            image.draw(in: CGRect(origin: .zero, size: image.size))
        }
    }
}
