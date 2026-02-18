//
//  PDFExportService.swift
//  DermFusion
//
//  Generates shareable scan report PDFs with cover page, scan details,
//  probabilities, optional GradCAM, and required medical disclaimer.
//

import Foundation
import UIKit

/// Generates shareable result reports as multi-page PDFs.
struct PDFExportService {

    private static let appName = "DermaFusion"
    private static let tagline = "Research & Educational Tool"
    private static let pageWidth: CGFloat = 612
    private static let pageHeight: CGFloat = 792
    private static let margin: CGFloat = 48
    private static let contentWidth: CGFloat = pageWidth - 2 * margin

    // MARK: - Public API

    /// Generates a PDF for the given scan record and returns the file URL.
    /// Caller is responsible for presenting the share sheet and cleaning up the temp file when done.
    func generatePDF(record: ScanRecord) throws -> URL {
        let format = UIGraphicsPDFRendererFormat()
        format.documentInfo = [
            (kCGPDFContextCreator as NSString) as String: Self.appName,
            (kCGPDFContextAuthor as NSString) as String: Self.appName,
            (kCGPDFContextTitle as NSString) as String: "\(Self.appName) Scan Report"
        ]
        let bounds = CGRect(x: 0, y: 0, width: Self.pageWidth, height: Self.pageHeight)
        let renderer = UIGraphicsPDFRenderer(bounds: bounds, format: format)

        let data = renderer.pdfData { context in
            context.beginPage()
            drawCoverPage(in: context.cgContext, record: record)
            context.beginPage()
            drawScanDetailsPage(in: context.cgContext, record: record)
            context.beginPage()
            drawProbabilitiesAndDisclaimerPage(in: context.cgContext, record: record)
        }

        let fileDateFormatter = DateFormatter()
        fileDateFormatter.dateFormat = "yyyy-MM-dd_HHmmss"
        let fileName = "DermaFusion_Scan_\(fileDateFormatter.string(from: record.timestamp)).pdf"
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(fileName)
        try data.write(to: tempURL)
        return tempURL
    }

    // MARK: - Cover Page

    private func drawCoverPage(in ctx: CGContext, record: ScanRecord) {
        let logoSize: CGFloat = 88
        let logoY = Self.margin + 60
        let logoImage = UIImage(named: "icon")

        if let logo = logoImage {
            let aspect = logo.size.width / logo.size.height
            let drawW = min(logoSize, Self.contentWidth)
            let drawH = drawW / aspect
            let rect = CGRect(
                x: Self.margin + (Self.contentWidth - drawW) / 2,
                y: logoY,
                width: drawW,
                height: min(drawH, logoSize)
            )
            logo.draw(in: rect)
        } else {
            let circleRect = CGRect(
                x: Self.margin + (Self.contentWidth - logoSize) / 2,
                y: logoY,
                width: logoSize,
                height: logoSize
            )
            ctx.setFillColor(UIColor.systemTeal.withAlphaComponent(0.2).cgColor)
            ctx.fillEllipse(in: circleRect)
            let attr: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 28, weight: .semibold),
                .foregroundColor: UIColor.systemTeal
            ]
            let str = "DF"
            (str as NSString).draw(
                at: CGPoint(x: circleRect.midX - (str as NSString).size(withAttributes: attr).width / 2,
                            y: circleRect.midY - (str as NSString).size(withAttributes: attr).height / 2),
                withAttributes: attr
            )
        }

        var y = logoY + logoSize + 32
        let titleAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 32, weight: .bold),
            .foregroundColor: UIColor.black
        ]
        let titleSize = (Self.appName as NSString).size(withAttributes: titleAttr)
        (Self.appName as NSString).draw(
            at: CGPoint(x: Self.margin + (Self.contentWidth - titleSize.width) / 2, y: y),
            withAttributes: titleAttr
        )
        y += titleSize.height + 8

        let subtitleAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 18, weight: .medium),
            .foregroundColor: UIColor.black
        ]
        let subtitle = "Scan Report"
        let subtitleSize = (subtitle as NSString).size(withAttributes: subtitleAttr)
        (subtitle as NSString).draw(
            at: CGPoint(x: Self.margin + (Self.contentWidth - subtitleSize.width) / 2, y: y),
            withAttributes: subtitleAttr
        )
        y += subtitleSize.height + 24

        let taglineAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 12, weight: .regular),
            .foregroundColor: UIColor.black
        ]
        let taglineSize = (Self.tagline as NSString).size(withAttributes: taglineAttr)
        (Self.tagline as NSString).draw(
            at: CGPoint(x: Self.margin + (Self.contentWidth - taglineSize.width) / 2, y: y),
            withAttributes: taglineAttr
        )
        y += taglineSize.height + 40

        let dateFormatter = DateFormatter()
        dateFormatter.dateStyle = .long
        dateFormatter.timeStyle = .short
        let dateStr = "Scanned \(dateFormatter.string(from: record.timestamp))"
        let dateAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 14, weight: .regular),
            .foregroundColor: UIColor.black
        ]
        let dateSize = (dateStr as NSString).size(withAttributes: dateAttr)
        (dateStr as NSString).draw(
            at: CGPoint(x: Self.margin + (Self.contentWidth - dateSize.width) / 2, y: y),
            withAttributes: dateAttr
        )
    }

    // MARK: - Scan Details Page

    private func drawScanDetailsPage(in ctx: CGContext, record: ScanRecord) {
        var y = Self.margin

        let sectionFont = UIFont.systemFont(ofSize: 20, weight: .bold)
        let sectionAttr: [NSAttributedString.Key: Any] = [
            .font: sectionFont,
            .foregroundColor: UIColor.black
        ]
        ("Scan Details" as NSString).draw(at: CGPoint(x: Self.margin, y: y), withAttributes: sectionAttr)
        y += sectionFont.lineHeight + 20

        if let image = UIImage(data: record.imageData), !record.imageData.isEmpty {
            let maxImageHeight: CGFloat = 280
            let aspect = image.size.width / image.size.height
            var drawWidth = Self.contentWidth
            var drawHeight = drawWidth / aspect
            if drawHeight > maxImageHeight {
                drawHeight = maxImageHeight
                drawWidth = drawHeight * aspect
            }
            let imageRect = CGRect(x: Self.margin + (Self.contentWidth - drawWidth) / 2, y: y, width: drawWidth, height: drawHeight)
            image.draw(in: imageRect)
            y += drawHeight + 24
        }

        let labelAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 11, weight: .medium),
            .foregroundColor: UIColor.black
        ]
        let valueAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 14, weight: .regular),
            .foregroundColor: UIColor.black
        ]

        let rows: [(String, String)] = [
            ("Primary assessment", record.primaryDiagnosis),
            ("Confidence", "\(Int((record.predictions.values.max() ?? 0) * 1000) / 10)%"),
            ("Risk level", (RiskLevel(rawValue: record.riskLevel) ?? .moderate).displayName),
            ("Location", record.lesionLocation),
            ("Sex", record.sex),
            ("Age", record.age.map { "\($0)" } ?? "—"),
            ("Date & time", formatTimestamp(record.timestamp))
        ]

        for (label, value) in rows {
            (label as NSString).draw(at: CGPoint(x: Self.margin, y: y), withAttributes: labelAttr)
            let valueSize = (value as NSString).size(withAttributes: valueAttr)
            (value as NSString).draw(
                at: CGPoint(x: Self.margin + Self.contentWidth - valueSize.width, y: y),
                withAttributes: valueAttr
            )
            y += max(18, valueSize.height) + 6
        }

        y += 16
        let bannerAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 11, weight: .regular),
            .foregroundColor: UIColor.black
        ]
        let bannerText = "Research tool only — not a medical diagnosis. Always consult a dermatologist."
        let bannerRect = CGRect(x: Self.margin, y: y, width: Self.contentWidth, height: 60)
        (bannerText as NSString).draw(in: bannerRect, withAttributes: bannerAttr)
    }

    // MARK: - Probabilities & Disclaimer Page

    private func drawProbabilitiesAndDisclaimerPage(in ctx: CGContext, record: ScanRecord) {
        var y = Self.margin

        let sectionFont = UIFont.systemFont(ofSize: 20, weight: .bold)
        let sectionAttr: [NSAttributedString.Key: Any] = [
            .font: sectionFont,
            .foregroundColor: UIColor.black
        ]
        ("Classification Probabilities" as NSString).draw(at: CGPoint(x: Self.margin, y: y), withAttributes: sectionAttr)
        y += sectionFont.lineHeight + 16

        let probabilityRows = LesionCategory.allCases.map { category in
            (category.displayName, record.probability(for: category))
        }.sorted { $0.1 > $1.1 }

        let headerAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 11, weight: .semibold),
            .foregroundColor: UIColor.black
        ]
        ("Lesion type" as NSString).draw(at: CGPoint(x: Self.margin, y: y), withAttributes: headerAttr)
        ("Probability" as NSString).draw(at: CGPoint(x: Self.margin + Self.contentWidth - 70, y: y), withAttributes: headerAttr)
        y += 20

        let rowFont = UIFont.systemFont(ofSize: 13, weight: .regular)
        let rowAttr: [NSAttributedString.Key: Any] = [
            .font: rowFont,
            .foregroundColor: UIColor.black
        ]
        let pctAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.monospacedDigitSystemFont(ofSize: 13, weight: .medium),
            .foregroundColor: UIColor.black
        ]

        for (name, prob) in probabilityRows {
            (name as NSString).draw(at: CGPoint(x: Self.margin, y: y), withAttributes: rowAttr)
            let pctStr = "\(Int(prob * 1000) / 10)%"
            let pctSize = (pctStr as NSString).size(withAttributes: pctAttr)
            (pctStr as NSString).draw(at: CGPoint(x: Self.margin + Self.contentWidth - pctSize.width, y: y), withAttributes: pctAttr)
            y += rowFont.lineHeight + 4
        }

        y += 24

        if let gradcamData = record.gradcamImageData, !gradcamData.isEmpty, let gradcamImage = UIImage(data: gradcamData) {
            let gradcamTitleAttr: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 14, weight: .semibold),
                .foregroundColor: UIColor.black
            ]
            ("GradCAM overlay (model attention)" as NSString).draw(at: CGPoint(x: Self.margin, y: y), withAttributes: gradcamTitleAttr)
            y += 22
            let maxGradCAMHeight: CGFloat = 180
            let aspect = gradcamImage.size.width / gradcamImage.size.height
            var w = Self.contentWidth
            var h = w / aspect
            if h > maxGradCAMHeight {
                h = maxGradCAMHeight
                w = h * aspect
            }
            gradcamImage.draw(in: CGRect(x: Self.margin + (Self.contentWidth - w) / 2, y: y, width: w, height: h))
            y += h + 24
        } else {
            let noGradCAMAttr: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 11, weight: .regular),
                .foregroundColor: UIColor.black
            ]
            ("GradCAM overlay not available for this scan." as NSString).draw(
                at: CGPoint(x: Self.margin, y: y),
                withAttributes: noGradCAMAttr
            )
            y += 20
        }

        if let notes = record.userNotes, !notes.isEmpty {
            let notesTitleAttr: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 14, weight: .semibold),
                .foregroundColor: UIColor.black
            ]
            ("Your notes" as NSString).draw(at: CGPoint(x: Self.margin, y: y), withAttributes: notesTitleAttr)
            y += 20
            let notesAttr: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 12, weight: .regular),
                .foregroundColor: UIColor.black
            ]
            let notesRect = CGRect(x: Self.margin, y: y, width: Self.contentWidth, height: 80)
            (notes as NSString).draw(in: notesRect, withAttributes: notesAttr)
            y += 90
        }

        y += 16
        let disclaimerTitleAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 12, weight: .semibold),
            .foregroundColor: UIColor.black
        ]
        ("Medical disclaimer" as NSString).draw(at: CGPoint(x: Self.margin, y: y), withAttributes: disclaimerTitleAttr)
        y += 18

        let disclaimerAttr: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 10, weight: .regular),
            .foregroundColor: UIColor.black
        ]
        let disclaimerRect = CGRect(x: Self.margin, y: y, width: Self.contentWidth, height: 120)
        (DFMedicalDisclaimer.fullText as NSString).draw(in: disclaimerRect, withAttributes: disclaimerAttr)
    }

    private func formatTimestamp(_ date: Date) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter.string(from: date)
    }
}
