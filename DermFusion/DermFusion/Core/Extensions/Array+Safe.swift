//
//  Array+Safe.swift
//  DermFusion
//
//  Safe subscript extension that prevents out-of-bounds crashes.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import Foundation

extension Array {
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
