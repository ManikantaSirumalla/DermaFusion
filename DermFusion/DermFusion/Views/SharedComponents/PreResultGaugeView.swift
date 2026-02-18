//
//  PreResultGaugeView.swift
//  DermFusion
//
//  Semi-circle (speedometer-style) gauge showing overall confidence with colored arc,
//  Low/High labels, numeric ticks, needle, and message. Used in the Pre-result section.
//

import SwiftUI
import Darwin

/// Confidence band for gauge message. Matches arc: green (low) → orange → red (high).
private enum GaugeConfidenceBand {
    case low      // 0..<0.5  → green  → recommend additional examination
    case medium   // 0.5..<0.75 → orange → consider clinical evaluation
    case high     // 0.75...1.0 → red   → high confidence

    static func from(_ value: Double) -> GaugeConfidenceBand {
        switch value {
        case ..<0.5: return .low
        case 0.5..<0.75: return .medium
        default: return .high
        }
    }

    var message: String {
        switch self {
        case .low: return "Recommend additional examination"
        case .medium: return "Consider clinical evaluation"
        case .high: return "High confidence detection"
        }
    }

    var messageColor: Color {
        switch self {
        case .low: return DFDesignSystem.Colors.riskLow
        case .medium: return DFDesignSystem.Colors.riskModerate
        case .high: return DFDesignSystem.Colors.riskHigh
        }
    }
}

/// Semi-circle gauge with colored arc (green / orange / red), Low/High labels,
/// 25/50/75 ticks, needle, and message text below.
struct PreResultGaugeView: View {

    /// Confidence in 0...1 (drives needle and message).
    let confidence: Double

    private let gaugeSize: CGFloat = 200
    private let lineWidth: CGFloat = 18
    private let needleLength: CGFloat = 68

    /// Upper semi-circle: position 0 → 180° (left), 0.5 → 270° (up), 1.0 → 360° (right).
    private func arcAngle(for position: Double) -> Angle {
        .degrees(180 + position * 180)
    }

    var body: some View {
        let clamped = min(max(confidence, 0), 1)
        let band = GaugeConfidenceBand.from(clamped)

        VStack(spacing: DFDesignSystem.Spacing.md) {
            ZStack {
                // ── Colored arc segments ──
                arcSegment(startFrac: 0.0, endFrac: 0.5, color: DFDesignSystem.Colors.riskLow)
                arcSegment(startFrac: 0.5, endFrac: 0.75, color: DFDesignSystem.Colors.riskModerate)
                arcSegment(startFrac: 0.75, endFrac: 1.0, color: DFDesignSystem.Colors.riskHigh)

                // ── Tick marks + labels at 25 / 50 / 75 ──
                tickMark(at: 0.25, label: "25")
                tickMark(at: 0.50, label: "50")
                tickMark(at: 0.75, label: "75")

                // ── Low / High endpoint labels ──
                endpointLabel("Low", at: 0.0)
                endpointLabel("High", at: 1.0)

                // ── Needle ──
                needleView(value: clamped)

                // ── Center percentage ──
                Text("\(Int(round(clamped * 100)))%")
                    .font(.title3)
                    .fontWeight(.bold)
                    .foregroundStyle(band.messageColor)
                    .offset(y: 10) // sits just below the needle pivot
            }
            .frame(width: gaugeSize + 60, height: gaugeSize / 2 + 40)

            // ── Message ──
            Text(band.message)
                .font(DFDesignSystem.Typography.caption)
                .fontWeight(.medium)
                .foregroundStyle(band.messageColor)
                .multilineTextAlignment(.center)
        }
    }

    // MARK: - Arc Segment

    /// Draws one colored portion of the semi-circle.
    private func arcSegment(startFrac: Double, endFrac: Double, color: Color) -> some View {
        let radius = gaugeSize / 2
        let startAngle = arcAngle(for: startFrac)
        let endAngle = arcAngle(for: endFrac)

        return Path { path in
            path.addArc(
                center: CGPoint(x: gaugeSize / 2 + 30, y: gaugeSize / 2 + 10),
                radius: radius,
                startAngle: startAngle,
                endAngle: endAngle,
                clockwise: false
            )
        }
        .stroke(color, style: StrokeStyle(lineWidth: lineWidth, lineCap: .round))
    }

    // MARK: - Tick Marks

    /// Small radial tick line + label outside the arc.
    private func tickMark(at fraction: Double, label: String) -> some View {
        let center = CGPoint(x: gaugeSize / 2 + 30, y: gaugeSize / 2 + 10)
        let radius = gaugeSize / 2
        let angle = arcAngle(for: fraction)
        let rad: Double = angle.radians

        let innerR = radius + lineWidth / 2 + 2
        let outerR = radius + lineWidth / 2 + 8
        let labelR = radius + lineWidth / 2 + 20

        let cosA = Darwin.cos(rad)
        let sinA = Darwin.sin(rad)

        return ZStack {
            // Tick line
            Path { path in
                path.move(to: CGPoint(x: center.x + innerR * CGFloat(cosA), y: center.y + innerR * CGFloat(sinA)))
                path.addLine(to: CGPoint(x: center.x + outerR * CGFloat(cosA), y: center.y + outerR * CGFloat(sinA)))
            }
            .stroke(DFDesignSystem.Colors.textSecondary.opacity(0.6), lineWidth: 1.5)

            // Label
            Text(label)
                .font(.system(size: 10, weight: .medium, design: .rounded))
                .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                .position(x: center.x + labelR * CGFloat(cosA), y: center.y + labelR * CGFloat(sinA))
        }
    }

    // MARK: - Endpoint Labels

    private func endpointLabel(_ text: String, at fraction: Double) -> some View {
        let center = CGPoint(x: gaugeSize / 2 + 30, y: gaugeSize / 2 + 10)
        let radius = gaugeSize / 2
        let labelR = radius + lineWidth / 2 + 22
        let angle = arcAngle(for: fraction)
        let rad: Double = angle.radians

        return Text(text)
            .font(.system(size: 11, weight: .semibold, design: .rounded))
            .foregroundStyle(DFDesignSystem.Colors.textSecondary)
            .position(
                x: center.x + labelR * CGFloat(Darwin.cos(rad)) + (fraction == 0 ? -10 : 10),
                y: center.y + labelR * CGFloat(Darwin.sin(rad))
            )
    }

    // MARK: - Needle

    private func needleView(value: Double) -> some View {
        let center = CGPoint(x: gaugeSize / 2 + 30, y: gaugeSize / 2 + 10)
        let angle = arcAngle(for: value)
        let rad: Double = angle.radians
        let cosR = CGFloat(Darwin.cos(rad))
        let sinR = CGFloat(Darwin.sin(rad))
        let innerRadius: CGFloat = 8
        let needleEnd = CGPoint(
            x: center.x + needleLength * cosR,
            y: center.y + needleLength * sinR
        )

        return ZStack {
            // Needle line
            Path { path in
                path.move(to: CGPoint(
                    x: center.x + innerRadius * cosR,
                    y: center.y + innerRadius * sinR
                ))
                path.addLine(to: needleEnd)
            }
            .stroke(
                DFDesignSystem.Colors.textPrimary,
                style: StrokeStyle(lineWidth: 2.5, lineCap: .round)
            )

            // Pivot dot
            Circle()
                .fill(DFDesignSystem.Colors.textPrimary)
                .frame(width: 10, height: 10)
                .position(center)
        }
    }
}

#Preview("Gauge") {
    VStack(spacing: 32) {
        PreResultGaugeView(confidence: 0.55)
        PreResultGaugeView(confidence: 0.2)
        PreResultGaugeView(confidence: 0.85)
    }
    .padding(DFDesignSystem.Spacing.lg)
}
