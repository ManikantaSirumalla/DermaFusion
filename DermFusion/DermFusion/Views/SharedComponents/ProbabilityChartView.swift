//
//  ProbabilityChartView.swift
//  DermFusion
//
//  Reusable probability distribution chart view.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Charts
import SwiftUI

/// Horizontal bar chart for lesion class probabilities with rounded bars and clear labels.
struct ProbabilityChartView: View {
    let probabilities: [(category: LesionCategory, probability: Double)]

    private static let barHeight: CGFloat = 20
    private static let barSpacing: CGFloat = 12
    private static let cornerRadius: CGFloat = 6

    var body: some View {
        Chart(probabilities, id: \.category) { item in
            BarMark(
                x: .value("Probability", item.probability * 100),
                y: .value("Category", item.category.displayName),
                width: .fixed(Self.barHeight)
            )
            .foregroundStyle(
                LinearGradient(
                    colors: [
                        DFDesignSystem.Colors.chartColor(for: item.category),
                        DFDesignSystem.Colors.chartColor(for: item.category).opacity(0.75)
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
            )
            .cornerRadius(Self.cornerRadius)
        }
        .chartXAxis {
            AxisMarks(values: .automatic(desiredCount: 5)) { value in
                AxisGridLine(stroke: StrokeStyle(lineWidth: 0.5))
                    .foregroundStyle(DFDesignSystem.Colors.divider.opacity(0.6))
                AxisValueLabel {
                    if let v = value.as(Double.self) {
                        Text("\(Int(v))%")
                            .font(.caption2)
                            .foregroundStyle(DFDesignSystem.Colors.textSecondary)
                    }
                }
            }
        }
        .chartYAxis {
            AxisMarks { value in
                AxisValueLabel {
                    if let name = value.as(String.self) {
                        Text(name)
                            .font(.caption)
                            .fontWeight(.medium)
                            .foregroundStyle(DFDesignSystem.Colors.textPrimary)
                    }
                }
            }
        }
        .chartXScale(domain: 0...100)
        .frame(height: CGFloat(probabilities.count) * (Self.barHeight + Self.barSpacing) + 44)
    }
}

#Preview {
    ProbabilityChartView(probabilities: [
        (.nevus, 0.87),
        (.melanoma, 0.04),
        (.bcc, 0.03),
        (.bkl, 0.025),
        (.akiec, 0.017),
        (.dermatofibroma, 0.009),
        (.vascular, 0.006),
    ])
    .padding(DFDesignSystem.Spacing.md)
}
