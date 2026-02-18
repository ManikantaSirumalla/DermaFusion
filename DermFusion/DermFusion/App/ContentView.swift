//
//  ContentView.swift
//  DermFusion
//
//  Root tab navigation for Scan, History, Learn, and About.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import SwiftUI

/// Root navigation surface for the DermaFusion app.
struct ContentView: View {

    // MARK: - Dependencies

    @EnvironmentObject private var appDataStore: AppDataStore

    // MARK: - Body

    var body: some View {
        TabView(selection: $appDataStore.selectedTab) {
            NavigationStack {
                ScanHomeView()
            }
            .tabItem {
                Label("Scan", systemImage: DFDesignSystem.Icons.tabScan)
            }
            .tag(AppTab.scan)

            NavigationStack {
                HistoryListView()
            }
            .tabItem {
                Label("History", systemImage: DFDesignSystem.Icons.tabHistory)
            }
            .tag(AppTab.history)

            NavigationStack {
                LearnView()
            }
            .tabItem {
                Label("Learn", systemImage: DFDesignSystem.Icons.tabLearn)
            }
            .tag(AppTab.learn)

            NavigationStack {
                AboutView()
            }
            .tabItem {
                Label("About", systemImage: DFDesignSystem.Icons.tabAbout)
            }
            .tag(AppTab.about)
        }
        .tint(DFDesignSystem.Colors.brandPrimary)
    }
}

#Preview("Default") {
    ContentView()
        .environmentObject(AppDataStore())
}

#Preview("Dark") {
    ContentView()
        .environmentObject(AppDataStore())
        .preferredColorScheme(.dark)
}
