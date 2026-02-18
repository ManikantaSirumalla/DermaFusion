//
//  HistoryListView.swift
//  DermFusion
//
//  History-tab root: saved scans in a native grouped list or empty state.
//

import SwiftUI

/// Displays local scan history and an empty state when no records are available.
struct HistoryListView: View {

    @EnvironmentObject private var appDataStore: AppDataStore

    var body: some View {
        Group {
            if appDataStore.records.isEmpty {
                emptyState
            } else {
                recordsList
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(DFDesignSystem.Colors.backgroundPrimary)
        .navigationTitle("History")
        .navigationBarTitleDisplayMode(.large)
        .showTabBarWhenRoot()
        .toolbar {
            if !appDataStore.records.isEmpty {
                Button("Delete All") {
                    appDataStore.clearAll()
                }
                .foregroundStyle(.red)
            }
        }
    }

    private var emptyState: some View {
        ContentUnavailableView {
            Label("No Scans Yet", systemImage: DFDesignSystem.Icons.tabHistory)
                .foregroundStyle(DFDesignSystem.Colors.brandPrimary)
        } description: {
            Text("Your saved analyses will appear here.")
        } actions: {
            Button("Start Your First Scan") {
                appDataStore.selectedTab = .scan
            }
            .buttonStyle(.borderedProminent)
            .tint(DFDesignSystem.Colors.brandPrimary)
        }
    }

    private var recordsList: some View {
        List {
            Section {
                ForEach(appDataStore.records, id: \.id) { record in
                    NavigationLink {
                        ScanDetailView(record: record)
                    } label: {
                        historyRow(record)
                    }
                }
                .onDelete { offsets in
                    for index in offsets {
                        appDataStore.delete(appDataStore.records[index])
                    }
                }
            } header: {
                Text("Scans")
            }
        }
        .listStyle(.insetGrouped)
    }

    private func historyRow(_ record: ScanRecord) -> some View {
        HStack(spacing: 12) {
            scanThumbnail(record)
            VStack(alignment: .leading, spacing: 2) {
                Text(record.primaryDiagnosis)
                    .font(.body)
                    .fontWeight(.medium)
                Text("\(Int((record.predictions.values.max() ?? 0) * 100))% confidence")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(record.lesionLocation)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text(record.timestamp.relativeDescription)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(.vertical, 4)
    }

    private func scanThumbnail(_ record: ScanRecord) -> some View {
        Group {
            if !record.imageData.isEmpty, let uiImage = UIImage(data: record.imageData) {
                Image(uiImage: uiImage)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 56, height: 56)
                    .clipped()
            } else {
                Image("raw")
                    .resizable()
                    .aspectRatio(contentMode: .fill)
                    .frame(width: 56, height: 56)
                    .clipped()
            }
        }
        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
    }
}

#Preview("With Records") {
    let store = AppDataStore()
    store.save(result: .sample, metadata: MetadataInput(age: 60, sex: .female, bodyRegion: .back))
    return NavigationStack { HistoryListView() }
        .environmentObject(store)
}

#Preview("Empty") {
    NavigationStack { HistoryListView() }
        .environmentObject(AppDataStore())
}

#Preview("Dark") {
    NavigationStack { HistoryListView() }
        .environmentObject(AppDataStore())
        .preferredColorScheme(.dark)
}
