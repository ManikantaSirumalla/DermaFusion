//
//  ImageReviewView.swift
//  DermFusion
//
//  Redesigned image review screen — full-bleed preview with floating controls
//  and a modern glass-material editing panel. Apple Photos / Camera style.
//

import SwiftUI
import UIKit

struct ImageReviewView: View {

    @Environment(\.dismiss) private var dismiss
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    let image: UIImage?
    let onRetake: (() -> Void)?

    // Navigation
    @State private var navigateToMetadata = false

    // Transform state
    @State private var scale: CGFloat = 1.0
    @State private var lastScale: CGFloat = 1.0
    @State private var offset: CGSize = .zero
    @State private var lastOffset: CGSize = .zero
    @State private var rotationDegrees: Double = 0

    // Crop (normalized 0–1 of image; circle inscribed in this rect)
    @State private var cropRect: CGRect = CGRect(x: 0.1, y: 0.1, width: 0.8, height: 0.8)
    @State private var isCropMode: Bool = false

    // Edit panel
    @State private var showEditTools: Bool = false

    init(image: UIImage? = nil, onRetake: (() -> Void)? = nil) {
        self.image = image
        self.onRetake = onRetake
    }

    var body: some View {
        ZStack {
            // Full-black canvas
            Color.black.ignoresSafeArea()

            // Image preview
            imageCanvas

            // Crop overlay
            if isCropMode {
                cropOverlay
            }

            // Floating UI layers (top bar only; bottom panel in safe area inset)
            VStack(spacing: 0) {
                topBar
                Spacer()
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            bottomPanel
        }
        .navigationBarHidden(true)
        .toolbar(.hidden, for: .navigationBar)
        .hideTabBarWhenPushed()
        .ignoresSafeArea(edges: .top)
        .statusBarHidden(false)
        .background(
            NavigationLink(
                destination: MetadataInputView(viewModel: ScanViewModel(capturedImage: processedImage)),
                isActive: $navigateToMetadata
            ) { EmptyView() }
                .hidden()
        )
    }
}

// MARK: - Top Bar

private extension ImageReviewView {

    var topBar: some View {
        HStack {
            // Back / Close
            floatingButton(icon: "chevron.left") {
                dismiss()
            }

            Spacer()

            // Image quality indicator
            if isLowResolutionImage {
                lowResIndicator
            }

            Spacer()

            // Reset (visible only when transforms applied)
            if canResetTransforms {
                floatingButton(icon: "arrow.counterclockwise") {
                    resetTransforms()
                }
                .transition(.opacity.combined(with: .scale(scale: 0.8)))
            }
        }
        .padding(.top, 56)
        .padding(.horizontal, DFDesignSystem.Spacing.md)
        .animation(.easeInOut(duration: 0.2), value: canResetTransforms)
    }

    var lowResIndicator: some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.orange)
            Text("Low Resolution")
                .font(.system(size: 12, weight: .semibold, design: .rounded))
                .foregroundStyle(.white.opacity(0.9))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 7)
        .background(
            Capsule(style: .continuous)
                .fill(.ultraThinMaterial)
        )
    }

    func floatingButton(icon: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            Image(systemName: icon)
                .font(.system(size: 15, weight: .semibold))
                .foregroundStyle(.white)
                .frame(width: 40, height: 40)
                .background(
                    Circle()
                        .fill(.ultraThinMaterial)
                )
        }
    }
}

// MARK: - Image Canvas

private extension ImageReviewView {

    var imageCanvas: some View {
        GeometryReader { geometry in
            Group {
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                        .rotationEffect(.degrees(rotationDegrees))
                } else {
                    emptyImagePlaceholder
                }
            }
            .frame(width: geometry.size.width, height: geometry.size.height)
            .scaleEffect(scale)
            .offset(offset)
            .gesture(combinedGesture)
        }
        .clipped()
    }

    var emptyImagePlaceholder: some View {
        VStack(spacing: 12) {
            Image(systemName: "photo.on.rectangle")
                .font(.system(size: 48, weight: .ultraLight))
                .foregroundStyle(.white.opacity(0.25))
            Text("No image selected")
                .font(.system(size: 14, weight: .medium, design: .rounded))
                .foregroundStyle(.white.opacity(0.3))
        }
    }

    var combinedGesture: some Gesture {
        SimultaneousGesture(
            MagnificationGesture()
                .onChanged { value in
                    scale = min(max(lastScale * value, 1.0), 4.0)
                }
                .onEnded { _ in
                    lastScale = scale
                    if scale <= 1.01 {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                            scale = 1.0
                            lastScale = 1.0
                            offset = .zero
                            lastOffset = .zero
                        }
                    }
                },
            DragGesture()
                .onChanged { value in
                    offset = CGSize(
                        width: lastOffset.width + value.translation.width,
                        height: lastOffset.height + value.translation.height
                    )
                }
                .onEnded { _ in
                    if scale <= 1.01 {
                        withAnimation(.spring(response: 0.3, dampingFraction: 0.8)) {
                            offset = .zero
                            lastOffset = .zero
                        }
                    } else {
                        lastOffset = offset
                    }
                }
        )
    }
}

// MARK: - Image frame (fitted + rotated AABB in view coords)

private extension ImageReviewView {

    /// Returns the axis-aligned bounding box of the fitted, rotated image in view coordinates.
    func imageFrameInView(viewSize: CGSize, imageSize: CGSize, rotationDegrees: Double) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else {
            return CGRect(origin: .zero, size: viewSize)
        }
        let scaleFit = min(
            viewSize.width / imageSize.width,
            viewSize.height / imageSize.height
        )
        let contentW = imageSize.width * scaleFit
        let contentH = imageSize.height * scaleFit
        let radians = CGFloat(rotationDegrees * .pi / 180)
        let cosR = abs(cos(radians))
        let sinR = abs(sin(radians))
        let aabbW = contentW * cosR + contentH * sinR
        let aabbH = contentW * sinR + contentH * cosR
        let cx = viewSize.width / 2
        let cy = viewSize.height / 2
        return CGRect(
            x: cx - aabbW / 2,
            y: cy - aabbH / 2,
            width: aabbW,
            height: aabbH
        )
    }
}

// MARK: - Crop Overlay (circular)

private extension ImageReviewView {

    var cropOverlay: some View {
        GeometryReader { geometry in
            let viewSize = geometry.size
            if let img = image {
                let imgSize = img.size
                let frame = imageFrameInView(
                    viewSize: viewSize,
                    imageSize: imgSize,
                    rotationDegrees: rotationDegrees
                )
                let diamNorm = min(cropRect.width, cropRect.height)
                let circleW = frame.width * diamNorm
                let circleH = frame.height * diamNorm
                let diameter = min(circleW, circleH)
                let centerX = frame.minX + frame.width * (cropRect.midX)
                let centerY = frame.minY + frame.height * (cropRect.midY)

                ZStack {
                    Color.black.opacity(0.6)
                        .ignoresSafeArea()

                    Circle()
                        .fill(Color.clear)
                        .frame(width: diameter, height: diameter)
                        .overlay(
                            Circle()
                                .strokeBorder(.white.opacity(0.9), lineWidth: 1.5)
                        )
                        .position(x: centerX, y: centerY)
                }
            }
        }
        .allowsHitTesting(false)
    }
}

// MARK: - Bottom Panel

private extension ImageReviewView {

    var bottomPanel: some View {
        VStack(spacing: 0) {
            if isCropMode {
                cropControls
            } else {
                editTools
                Divider()
                    .background(.white.opacity(0.1))
                actionButtons
            }
        }
        .padding(.top, DFDesignSystem.Spacing.sm)
        .padding(.bottom, 0)
        .background(
            Rectangle()
                .fill(.ultraThinMaterial)
                .ignoresSafeArea(edges: .bottom)
        )
    }
}

// MARK: - Edit Tools Row

private extension ImageReviewView {

    var editTools: some View {
        VStack(spacing: DFDesignSystem.Spacing.md) {
            // Tilt control
            tiltControl

            // Tool buttons
            HStack(spacing: 24) {
                editToolButton(icon: "crop", label: "Crop", isActive: false) {
                    withAnimation(.easeInOut(duration: 0.25)) {
                        scale = 1.0
                        lastScale = 1.0
                        offset = .zero
                        lastOffset = .zero
                        isCropMode = true
                    }
                }

                editToolButton(icon: "arrow.triangle.2.circlepath", label: "Rotate 90°", isActive: false) {
                    withAnimation(.spring(response: 0.3, dampingFraction: 0.7)) {
                        rotationDegrees += 90
                        if rotationDegrees >= 360 { rotationDegrees -= 360 }
                    }
                }
            }
        }
        .padding(.horizontal, DFDesignSystem.Spacing.lg)
        .padding(.top, DFDesignSystem.Spacing.md)
        .padding(.bottom, DFDesignSystem.Spacing.md)
    }

    var tiltControl: some View {
        VStack(spacing: 8) {
            HStack {
                Image(systemName: "level")
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.white.opacity(0.5))

                Text("Straighten")
                    .font(.system(size: 13, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.7))

                Spacer()

                Text("\(String(format: "%.1f", rotationDegrees))°")
                    .font(.system(size: 13, weight: .semibold, design: .monospaced))
                    .foregroundStyle(
                        abs(rotationDegrees) > 0.5
                            ? DFDesignSystem.Colors.brandPrimary
                            : .white.opacity(0.4)
                    )
            }

            // Custom tilt slider
            tiltSlider
        }
    }

    var tiltSlider: some View {
        GeometryReader { geo in
            let width = geo.size.width
            let tickCount = 31 // -15 to +15
            let tickSpacing = width / CGFloat(tickCount - 1)

            ZStack {
                // Tick marks
                HStack(spacing: 0) {
                    ForEach(0..<tickCount, id: \.self) { i in
                        let isMajor = i % 5 == 0
                        let isCenter = i == tickCount / 2
                        Rectangle()
                            .fill(
                                isCenter ? DFDesignSystem.Colors.brandPrimary :
                                    isMajor ? .white.opacity(0.4) : .white.opacity(0.15)
                            )
                            .frame(
                                width: isCenter ? 2 : 1,
                                height: isCenter ? 16 : (isMajor ? 12 : 8)
                            )
                        if i < tickCount - 1 {
                            Spacer(minLength: 0)
                        }
                    }
                }
                .frame(height: 16)

                // Slider (low opacity so custom track shows; full content shape for hit-testing)
                Slider(value: $rotationDegrees, in: -15...15, step: 0.5)
                    .tint(DFDesignSystem.Colors.brandPrimary)
                    .opacity(0.35)
                    .frame(height: 44)
                    .contentShape(Rectangle())

                // Indicator line
                let normalizedPos = (rotationDegrees + 15) / 30
                let indicatorX = width * CGFloat(normalizedPos)
                Rectangle()
                    .fill(DFDesignSystem.Colors.brandPrimary)
                    .frame(width: 2.5, height: 22)
                    .position(x: indicatorX, y: 8)
                    .allowsHitTesting(false)
            }
        }
        .frame(height: 28)
    }

    func editToolButton(icon: String, label: String, isActive: Bool, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 6) {
                Image(systemName: icon)
                    .font(.system(size: 20, weight: .medium))
                    .foregroundStyle(isActive ? DFDesignSystem.Colors.brandPrimary : .white.opacity(0.85))
                    .frame(width: 44, height: 44)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(
                                isActive
                                    ? DFDesignSystem.Colors.brandPrimary.opacity(0.15)
                                    : .white.opacity(0.08)
                            )
                    )

                Text(label)
                    .font(.system(size: 11, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.6))
            }
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Crop Controls

private extension ImageReviewView {

    var cropControls: some View {
        HStack(spacing: DFDesignSystem.Spacing.md) {
            Button {
                withAnimation(.easeInOut(duration: 0.25)) {
                    cropRect = CGRect(x: 0.1, y: 0.1, width: 0.8, height: 0.8)
                    isCropMode = false
                }
            } label: {
                Text("Cancel")
                    .font(.system(size: 15, weight: .medium, design: .rounded))
                    .foregroundStyle(.white.opacity(0.8))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }

            Button {
                withAnimation(.easeInOut(duration: 0.25)) {
                    isCropMode = false
                }
            } label: {
                Text("Apply")
                    .font(.system(size: 15, weight: .bold, design: .rounded))
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(DFDesignSystem.Colors.brandPrimary)
                    )
            }
        }
        .buttonStyle(.plain)
        .padding(.horizontal, DFDesignSystem.Spacing.lg)
        .padding(.vertical, DFDesignSystem.Spacing.md)
    }
}

// MARK: - Action Buttons

private extension ImageReviewView {

    var actionButtons: some View {
        HStack(spacing: DFDesignSystem.Spacing.md) {
            // Retake
            Button {
                resetTransforms()
                onRetake?() ?? dismiss()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "camera")
                        .font(.system(size: 14, weight: .semibold))
                    Text("Retake")
                        .font(.system(size: 15, weight: .semibold, design: .rounded))
                }
                .foregroundStyle(.white.opacity(0.9))
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(.white.opacity(0.1))
                        .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                )
            }
            .buttonStyle(.plain)

            // Continue (no glow)
            Button {
                navigateToMetadata = true
            } label: {
                HStack(spacing: 6) {
                    Text("Continue")
                        .font(.system(size: 15, weight: .bold, design: .rounded))
                    Image(systemName: "arrow.right")
                        .font(.system(size: 13, weight: .bold))
                }
                .foregroundStyle(.white)
                .frame(maxWidth: .infinity)
                .padding(.vertical, 16)
                .background(
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .fill(DFDesignSystem.Colors.brandPrimary)
                )
            }
            .buttonStyle(.plain)
        }
        .padding(.horizontal, DFDesignSystem.Spacing.lg)
        .padding(.top, DFDesignSystem.Spacing.md)
        .padding(.bottom, DFDesignSystem.Spacing.sm)
    }
}

// MARK: - Logic

private extension ImageReviewView {

    var processedImage: UIImage? {
        guard let image else { return nil }
        var current = image
        if abs(rotationDegrees) > 0.5 {
            current = current.rotated(degrees: rotationDegrees) ?? current
        }
        let r = cropRect
        let isFullImage = r.minX <= 0.01 && r.minY <= 0.01 && r.width >= 0.99 && r.height >= 0.99
        if !isFullImage {
            current = current.croppedToCircle(normalizedRect: r) ?? current
        }
        return current
    }

    var isLowResolutionImage: Bool {
        guard let image else { return false }
        return image.size.width < 200 || image.size.height < 200
    }

    func resetTransforms() {
        withAnimation(.spring(response: 0.35, dampingFraction: 0.8)) {
            scale = 1.0
            lastScale = 1.0
            offset = .zero
            lastOffset = .zero
            rotationDegrees = 0
            cropRect = CGRect(x: 0.1, y: 0.1, width: 0.8, height: 0.8)
            isCropMode = false
        }
    }

    var canResetTransforms: Bool {
        abs(scale - 1.0) > 0.01
            || abs(offset.width) > 0.5
            || abs(offset.height) > 0.5
            || abs(rotationDegrees) > 0.5
            || cropRect.minX > 0.02 || cropRect.minY > 0.02
            || cropRect.width < 0.98 || cropRect.height < 0.98
    }
}

// MARK: - UIImage Helpers

private extension UIImage {
    func rotated(degrees: Double) -> UIImage? {
        let radians = CGFloat(degrees * .pi / 180)
        let newSize = CGRect(origin: .zero, size: size)
            .applying(CGAffineTransform(rotationAngle: radians))
            .integral.size
        let renderer = UIGraphicsImageRenderer(size: newSize)
        return renderer.image { context in
            let ctx = context.cgContext
            ctx.translateBy(x: newSize.width / 2, y: newSize.height / 2)
            ctx.rotate(by: radians)
            ctx.translateBy(x: -size.width / 2, y: -size.height / 2)
            draw(at: .zero)
        }
    }

    /// Crops to the normalized rect then masks to an inscribed circle (transparent outside).
    func croppedToCircle(normalizedRect: CGRect) -> UIImage? {
        guard let cg = cgImage else { return nil }
        let w = CGFloat(cg.width)
        let h = CGFloat(cg.height)
        let rect = CGRect(
            x: max(0, normalizedRect.origin.x * w),
            y: max(0, normalizedRect.origin.y * h),
            width: min(normalizedRect.width * w, w - normalizedRect.origin.x * w),
            height: min(normalizedRect.height * h, h - normalizedRect.origin.y * h)
        )
        guard rect.width > 0, rect.height > 0,
              let cropped = cg.cropping(to: rect) else { return nil }
        let size = rect.size
        let renderer = UIGraphicsImageRenderer(size: size)
        let radius = min(size.width, size.height) / 2
        let center = CGPoint(x: size.width / 2, y: size.height / 2)
        let masked = renderer.image { context in
            let ctx = context.cgContext
            ctx.saveGState()
            ctx.addEllipse(in: CGRect(x: center.x - radius, y: center.y - radius,
                                      width: radius * 2, height: radius * 2))
            ctx.clip()
            UIImage(cgImage: cropped, scale: scale, orientation: imageOrientation)
                .draw(in: CGRect(origin: .zero, size: size))
            ctx.restoreGState()
        }
        return masked
    }
}

// MARK: - Previews

#Preview("With Image") {
    ImageReviewView(image: nil, onRetake: nil)
}

#Preview("Dark") {
    ImageReviewView(image: nil, onRetake: nil)
        .preferredColorScheme(.dark)
}
