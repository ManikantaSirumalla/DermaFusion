//
//  AboutOverviewView.swift
//  DermFusion
//
//  Intended use and scope details. Native List with grouped sections.
//

import SwiftUI

struct AboutOverviewView: View {

    var body: some View {
        List {
            Section {
                VStack(alignment: .leading, spacing: 8) {
                    Text("DermaFusion is an end-to-end portfolio project that takes a research-grade multimodal model from training to a polished iOS experience.")
                        .font(.body)
                        .foregroundStyle(.primary)
                    Text("The app combines dermoscopic image analysis with clinical metadata (age, sex, lesion location) to produce a classification result for educational use.")
                        .font(.body)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            } header: {
                Text("About")
            }

            Section {
                listBullet("Data and training pipeline with strict lesion-level leakage controls.")
                listBullet("Multimodal model development and evaluation for 7 target lesion classes.")
                listBullet("CoreML deployment strategy for on-device inference.")
                listBullet("Native iOS product design: Scan, History, Learn, and About experiences.")
                listBullet("Offline educational content system with source attribution and references.")
            } header: {
                Text("Project Scope")
            }

            Section {
                listBullet("Privacy first: all sensitive image and metadata processing stays on device.")
                listBullet("Clinical responsibility: language is educational and never diagnostic.")
                listBullet("Reliability over novelty: offline-first content and deterministic core flows.")
                listBullet("Traceability: each major feature ties back to design-doc requirements.")
                listBullet("User clarity: risk communication uses text + icon + color, never color only.")
            } header: {
                Text("How Decisions Are Made")
            }

            Section {
                Text("The app is currently in UI/UX and integration phase. CoreML inference and parity validation are integrated after model checkpoints are finalized.")
                    .font(.body)
                    .foregroundStyle(.secondary)
            } header: {
                Text("Current Phase")
            } footer: {
                Text("DermaFusion is a research and educational tool and does not provide medical diagnosis.")
                    .font(.caption)
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Overview")
        .navigationBarTitleDisplayMode(.inline)
        .hideTabBarWhenPushed()
    }

    private func listBullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 8) {
            Text("•")
                .foregroundStyle(.secondary)
            Text(text)
                .font(.body)
                .foregroundStyle(.primary)
        }
        .padding(.vertical, 2)
    }
}

#Preview {
    NavigationStack { AboutOverviewView() }
}
