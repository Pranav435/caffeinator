// Draws Resources/AppIcon.icns: a white coffee cup on an amber tile.
// Run from the repo root: swift scripts/make-icon.swift
import AppKit

func draw(_ px: Int) -> Data {
    let s = CGFloat(px) / 1024
    let ctx = CGContext(data: nil, width: px, height: px, bitsPerComponent: 8, bytesPerRow: 0,
                        space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.scaleBy(x: s, y: s)

    // Tile with Apple's 100 pt margin and a soft shadow.
    let tile = CGPath(roundedRect: CGRect(x: 100, y: 100, width: 824, height: 824), cornerWidth: 185, cornerHeight: 185, transform: nil)
    ctx.saveGState()
    ctx.setShadow(offset: CGSize(width: 0, height: -10), blur: 24, color: CGColor(gray: 0, alpha: 0.3))
    ctx.addPath(tile)
    ctx.setFillColor(CGColor(gray: 0, alpha: 1))
    ctx.fillPath()
    ctx.restoreGState()
    ctx.saveGState()
    ctx.addPath(tile)
    ctx.clip()
    let gradient = CGGradient(colorsSpace: nil, colors: [
        CGColor(red: 0.99, green: 0.67, blue: 0.27, alpha: 1), CGColor(red: 0.80, green: 0.33, blue: 0.10, alpha: 1),
    ] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(gradient, start: CGPoint(x: 512, y: 924), end: CGPoint(x: 512, y: 100), options: [])
    ctx.restoreGState()

    ctx.setFillColor(.white)
    ctx.setStrokeColor(.white)
    ctx.setLineCap(.round)

    // Saucer
    ctx.fillEllipse(in: CGRect(x: 250, y: 238, width: 524, height: 64))
    // Cup body: flat top, rounded bottom
    let cup = CGMutablePath()
    cup.move(to: CGPoint(x: 300, y: 600))
    cup.addLine(to: CGPoint(x: 680, y: 600))
    cup.addCurve(to: CGPoint(x: 560, y: 300), control1: CGPoint(x: 680, y: 440), control2: CGPoint(x: 640, y: 320))
    cup.addLine(to: CGPoint(x: 420, y: 300))
    cup.addCurve(to: CGPoint(x: 300, y: 600), control1: CGPoint(x: 340, y: 320), control2: CGPoint(x: 300, y: 440))
    cup.closeSubpath()
    ctx.addPath(cup)
    ctx.fillPath()
    // Handle
    ctx.setLineWidth(38)
    ctx.addArc(center: CGPoint(x: 690, y: 500), radius: 62, startAngle: .pi * 0.62, endAngle: -.pi * 0.62, clockwise: true)
    ctx.strokePath()
    // Steam
    ctx.setLineWidth(30)
    for x in [410.0, 490, 570] {
        ctx.move(to: CGPoint(x: x, y: 660))
        ctx.addCurve(to: CGPoint(x: x, y: 820), control1: CGPoint(x: x - 45, y: 715), control2: CGPoint(x: x + 45, y: 765))
    }
    ctx.strokePath()

    return NSBitmapImageRep(cgImage: ctx.makeImage()!).representation(using: .png, properties: [:])!
}

let iconset = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("AppIcon.iconset")
try? FileManager.default.removeItem(at: iconset)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    try draw(size).write(to: iconset.appendingPathComponent("icon_\(size)x\(size).png"))
    try draw(size * 2).write(to: iconset.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
}
let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconset.path, "-o", "Resources/AppIcon.icns"]
try iconutil.run()
iconutil.waitUntilExit()
print(iconutil.terminationStatus == 0 ? "Wrote Resources/AppIcon.icns" : "iconutil failed")
