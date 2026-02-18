//
//  Date+Formatting.swift
//  DermFusion
//
//  Date formatting helpers for relative and absolute display strings.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

extension Date {
    var relativeDescription: String {
        RelativeDateTimeFormatter().localizedString(for: self, relativeTo: Date())
    }
}
