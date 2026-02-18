//
//  LearnView.swift
//  DermFusion
//
//  Educational hub with ABCDE, lesion types, and dermoscopy. Native grouped list.
//

import SwiftUI

/// Educational area for lesion learning resources.
struct LearnView: View {

    @StateObject private var viewModel = EducationViewModel()

    var body: some View {
        List {
            if let loadError = viewModel.loadError {
                Section {
                    Label {
                        Text("Using bundled fallback educational content. \(loadError)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    } icon: {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                            .symbolRenderingMode(.hierarchical)
                    }
                }
            }

            Section {
                NavigationLink {
                    ABCDERuleView(points: viewModel.abcdePoints)
                } label: {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("The ABCDE Rule")
                                .font(.body)
                                .fontWeight(.medium)
                            Text(viewModel.abcdePoints.first?.description ?? "Five warning signs to watch for.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                    } icon: {
                        Image(systemName: "list.bullet.rectangle.fill")
                            .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                            .symbolRenderingMode(.hierarchical)
                    }
                }
            } header: {
                Text("Screening")
            }

            Section {
                ForEach(viewModel.lesions) { lesion in
                    NavigationLink {
                        LesionTypeDetailView(content: lesion)
                    } label: {
                        HStack(spacing: 12) {
                            lesionThumbnail(lesion)
                            VStack(alignment: .leading, spacing: 2) {
                                Text(lesion.title)
                                    .font(.body)
                                    .fontWeight(.medium)
                                Text(lesion.severity)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            } header: {
                Text("Skin Lesion Types")
            }

            Section {
                NavigationLink {
                    DermoscopyBasicsView(paragraphs: viewModel.dermoscopyParagraphs)
                } label: {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("What is Dermoscopy?")
                                .font(.body)
                                .fontWeight(.medium)
                            Text(viewModel.dermoscopyParagraphs.first?.body ?? "Understand how dermoscopic images support classification.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)
                        }
                    } icon: {
                        Image(systemName: "magnifyingglass")
                            .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                            .symbolRenderingMode(.hierarchical)
                    }
                }
            } header: {
                Text("Basics")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Learn")
        .navigationBarTitleDisplayMode(.large)
        .showTabBarWhenRoot()
    }

    private func lesionThumbnail(_ lesion: LesionEducation) -> some View {
        let imageName = lesion.examples.first?.imageName ?? lesion.category?.assetImageName
        return Group {
            if let name = imageName {
                Image(name)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 44, height: 44)
                    .clipped()
            } else {
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .fill(DFDesignSystem.Colors.chartColor(for: lesion.category ?? .nevus).opacity(0.3))
                    .overlay {
                        Image(systemName: "photo")
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }
                    .frame(width: 44, height: 44)
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
    }
}

#Preview("Default") {
    NavigationStack { LearnView() }
}

#Preview("Dark") {
    NavigationStack { LearnView() }
        .preferredColorScheme(.dark)
}
