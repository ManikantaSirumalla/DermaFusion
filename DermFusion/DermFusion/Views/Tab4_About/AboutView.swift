//
//  AboutView.swift
//  DermFusion
//
//  Transparency screen with model, privacy, and disclaimer sections.
//  Native Apple Settings–style grouped list.
//

import SwiftUI

/// Root about screen using Apple-style grouped navigation sections.
struct AboutView: View {

    var body: some View {
        List {
            Section {
                NavigationLink {
                    AboutOverviewView()
                } label: {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Overview")
                                .font(.body)
                                .fontWeight(.medium)
                            Text("Intended use and scope")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: DFDesignSystem.Icons.aboutOverview)
                            .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                            .font(.body)
                    }
                }

                NavigationLink {
                    AboutModelView()
                } label: {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Model Information")
                                .font(.body)
                                .fontWeight(.medium)
                            Text("Architecture and limitations")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: DFDesignSystem.Icons.aboutModel)
                            .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                            .font(.body)
                    }
                }

                NavigationLink {
                    AboutPrivacyView()
                } label: {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Privacy & Data")
                                .font(.body)
                                .fontWeight(.medium)
                            Text("How data stays on-device")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: DFDesignSystem.Icons.aboutPrivacy)
                            .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                            .font(.body)
                    }
                }

                NavigationLink {
                    AboutDisclaimerView()
                } label: {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("Medical Disclaimer")
                                .font(.body)
                                .fontWeight(.medium)
                            Text("Required educational notice")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: DFDesignSystem.Icons.disclaimer)
                            .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                            .font(.body)
                    }
                }

                NavigationLink {
                    AboutReferencesView()
                } label: {
                    Label {
                        VStack(alignment: .leading, spacing: 2) {
                            Text("References")
                                .font(.body)
                                .fontWeight(.medium)
                            Text("Educational data sources")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    } icon: {
                        Image(systemName: DFDesignSystem.Icons.aboutReferences)
                            .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
                            .font(.body)
                    }
                }
            } header: {
                Text("DermaFusion")
            }

            Section {
                HStack {
                    Text("Version")
                    Spacer()
                    Text("1.0.0 (1)")
                        .foregroundStyle(.secondary)
                }
                .font(.body)
            } header: {
                Text("App")
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("About")
        .navigationBarTitleDisplayMode(.large)
        .showTabBarWhenRoot()
    }
}

#Preview("Default") {
    NavigationStack { AboutView() }
}

#Preview("Dark") {
    NavigationStack { AboutView() }
        .preferredColorScheme(.dark)
}
