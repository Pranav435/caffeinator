// Draws the app icon (Resources/AppIcon.icns, docs/icon.png) and the GitHub social preview
// card (docs/social-preview.png) from one cup drawing.
// Run from the repo root: swift scripts/make-art.swift
import AppKit

func canvas(_ width: Int, _ height: Int) -> CGContext {
    CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
              space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
}

func png(_ ctx: CGContext) -> Data {
    NSBitmapImageRep(cgImage: ctx.makeImage()!).representation(using: .png, properties: [:])!
}

/// The icon, drawn in a 1024-point square whose bottom-left corner is at `origin`.
func drawIcon(_ ctx: CGContext, origin: CGPoint = .zero, size: CGFloat = 1024) {
    ctx.saveGState()
    ctx.translateBy(x: origin.x, y: origin.y)
    ctx.scaleBy(x: size / 1024, y: size / 1024)

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
    ctx.restoreGState()
}

func icon(_ px: Int) -> Data {
    let ctx = canvas(px, px)
    drawIcon(ctx, size: CGFloat(px))
    return png(ctx)
}

/// 1280×640 card GitHub shows when the repo link is shared.
func socialPreview() -> Data {
    let ctx = canvas(1280, 640)
    let background = CGGradient(colorsSpace: nil, colors: [
        CGColor(red: 0.20, green: 0.12, blue: 0.08, alpha: 1), CGColor(red: 0.10, green: 0.06, blue: 0.04, alpha: 1),
    ] as CFArray, locations: [0, 1])!
    ctx.drawLinearGradient(background, start: CGPoint(x: 0, y: 640), end: CGPoint(x: 1280, y: 0), options: [])
    drawIcon(ctx, origin: CGPoint(x: 70, y: 110), size: 420)

    NSGraphicsContext.current = NSGraphicsContext(cgContext: ctx, flipped: false)
    func font(_ size: CGFloat, _ weight: NSFont.Weight) -> NSFont {
        let base = NSFont.systemFont(ofSize: size, weight: weight)
        return NSFont(descriptor: base.fontDescriptor.withDesign(.rounded) ?? base.fontDescriptor, size: size) ?? base
    }
    let lines: [(String, NSFont, NSColor, CGFloat)] = [
        ("Caffeinator", font(104, .bold), .white, 380),
        ("Your Mac is a gifted napper.", font(40, .medium), NSColor(white: 1, alpha: 0.92), 300),
        ("This is the coffee.", font(40, .medium), NSColor(red: 0.99, green: 0.72, blue: 0.35, alpha: 1), 248),
        ("Free  ·  Open source  ·  Built for VoiceOver", font(27, .regular), NSColor(white: 1, alpha: 0.6), 170),
    ]
    for (text, font, color, y) in lines {
        NSAttributedString(string: text, attributes: [.font: font, .foregroundColor: color]).draw(at: CGPoint(x: 540, y: y))
    }
    NSGraphicsContext.current = nil
    return png(ctx)
}

let fm = FileManager.default
try fm.createDirectory(atPath: "docs", withIntermediateDirectories: true)
try icon(512).write(to: URL(fileURLWithPath: "docs/icon.png"))
try socialPreview().write(to: URL(fileURLWithPath: "docs/social-preview.png"))

let iconset = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("AppIcon.iconset")
try? fm.removeItem(at: iconset)
try fm.createDirectory(at: iconset, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    try icon(size).write(to: iconset.appendingPathComponent("icon_\(size)x\(size).png"))
    try icon(size * 2).write(to: iconset.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
}
let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconset.path, "-o", "Resources/AppIcon.icns"]
try iconutil.run()
iconutil.waitUntilExit()
print(iconutil.terminationStatus == 0 ? "Wrote Resources/AppIcon.icns, docs/icon.png, docs/social-preview.png" : "iconutil failed")
