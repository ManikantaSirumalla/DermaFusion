//
//  UIImage+Utilities.swift
//  DermFusion
//
//  UIImage utility helpers used by camera, review, and persistence features.
//
//  Created by Manikanta Sirumalla on 2/13/26.
//

import UIKit

extension UIImage {
    func jpegDataCompressed(quality: CGFloat = 0.7) -> Data? {
        jpegData(compressionQuality: quality)
    }
}
