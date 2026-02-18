//
//  View+TabBarVisibility.swift
//  DermFusion
//
//  Hides the tab bar on child screens so it only appears on tab root views.
//  Uses UIKit for reliable behavior on iOS 17.4+ where toolbar(.hidden, for: .tabBar) can fail.
//

import SwiftUI
import UIKit

extension View {

    /// Call on any pushed (child) view so the tab bar is hidden.
    func hideTabBarWhenPushed() -> some View {
        modifier(TabBarVisibilityModifier(hide: true))
    }

    /// Call on tab root views so the tab bar is shown when returning to them.
    func showTabBarWhenRoot() -> some View {
        modifier(TabBarVisibilityModifier(hide: false))
    }
}

private struct TabBarVisibilityModifier: ViewModifier {

    let hide: Bool

    func body(content: Content) -> some View {
        content
            .toolbar(hide ? .hidden : .visible, for: .tabBar)
            .onAppear { setTabBarHidden(hide) }
    }

    private func setTabBarHidden(_ hidden: Bool) {
        DispatchQueue.main.async {
            guard let windowScene = UIApplication.shared.connectedScenes
                .compactMap({ $0 as? UIWindowScene })
                .first(where: { $0.activationState == .foregroundActive }),
                  let window = windowScene.windows.first(where: \.isKeyWindow),
                  let tabBarController = window.rootViewController?.findTabBarController() else { return }
            tabBarController.tabBar.isHidden = hidden
        }
    }
}

private extension UIViewController {

    func findTabBarController() -> UITabBarController? {
        if let tab = self as? UITabBarController { return tab }
        for child in children {
            if let tab = child.findTabBarController() { return tab }
        }
        return nil
    }
}
