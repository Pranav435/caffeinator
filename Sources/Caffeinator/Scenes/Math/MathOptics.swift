/// Metal shared by the Math scenes, which are filmed as lab photographs: the colour of light at a wavelength, and
/// streaks of moving light as a camera records them through a real lens. Put it in front of a scene's source.
enum MathOptics {
    static let source = """
    // Linear sRGB of monochromatic light at `nm`, from Wyman, Sloan and Shirley's fit of the CIE 1931 observer, with unit
    // luminance. Out-of-gamut colours are desaturated toward white rather than clipped, the way a camera renders an LED.
    inline float3 wavelength(float nm) {
        float t1 = (nm - 442.0) * (nm < 442.0 ? 0.0624 : 0.0374);
        float t2 = (nm - 599.8) * (nm < 599.8 ? 0.0264 : 0.0323);
        float t3 = (nm - 501.1) * (nm < 501.1 ? 0.0490 : 0.0382);
        float x = 0.362 * exp(-0.5 * t1 * t1) + 1.056 * exp(-0.5 * t2 * t2) - 0.065 * exp(-0.5 * t3 * t3);
        float t4 = (nm - 568.8) * (nm < 568.8 ? 0.0213 : 0.0247);
        float t5 = (nm - 530.9) * (nm < 530.9 ? 0.0613 : 0.0322);
        float y = 0.821 * exp(-0.5 * t4 * t4) + 0.286 * exp(-0.5 * t5 * t5);
        float t6 = (nm - 437.0) * (nm < 437.0 ? 0.0845 : 0.0278);
        float t7 = (nm - 459.0) * (nm < 459.0 ? 0.0385 : 0.0725);
        float z = 1.217 * exp(-0.5 * t6 * t6) + 0.681 * exp(-0.5 * t7 * t7);
        float3 c = float3(3.2406 * x - 1.5372 * y - 0.4986 * z, -0.9689 * x + 1.8758 * y + 0.0415 * z,
                          0.0557 * x - 0.2040 * y + 1.0570 * z);
        c -= min(min(c.r, min(c.g, c.b)), 0.0);
        return c / max(dot(c, float3(0.2126, 0.7152, 0.0722)), 1e-4);
    }

    // The same colour scaled so its strongest channel is one: how a camera records a bright, saturated LED.
    inline float3 ledColor(float nm) {
        float3 c = wavelength(nm);
        return c / max(c.r, max(c.g, c.b));
    }

    // A quad around the segment a→b (pixels, y down), reaching `radius` beyond it on every side, for streaks of light.
    // `local` comes out in pixels: x along the segment from a, y across it.
    struct Streak { float4 pos [[position]]; float2 local; float len; float blur; float4 color; };

    inline Streak streakCorner(float2 a, float2 b, float blur, uint corner, float4 color, float2 res) {
        float2 d = b - a;
        float len = length(d);
        float2 dir = len > 1e-3 ? d / len : float2(1, 0);
        float2 nrm = float2(-dir.y, dir.x);
        float radius = max(blur + 1.5, 3.5 * sqrt(0.8 + 0.25 * blur * blur));
        const uint corners[6] = {0, 1, 2, 2, 1, 3};
        uint k = corners[corner];
        float along = (k & 1) ? len + radius : -radius;
        float across = (k & 2) ? radius : -radius;
        float2 s = a + dir * along + nrm * across;
        Streak o;
        o.pos = float4(s.x / res.x * 2 - 1, 1 - s.y / res.y * 2, 0, 1);
        o.local = float2(along, across);
        o.len = len;
        o.blur = blur;
        o.color = color;
        return o;
    }

    inline float normalCDF(float t) { return 1 / (1 + exp(-1.702 * t)); }

    // Share of a streak's light per pixel at `local`: a point of light crossed the segment at an even speed during the
    // exposure, so its light is spread along the path, faint where it was fast and bright where it was slow. In focus
    // the lens draws it as a small Gaussian spot; defocused by a blur disc of radius `blur` pixels it becomes the
    // flat-topped band with crisp shoulders that real bokeh makes of a moving light. Integrates to one.
    inline float streakLight(float2 q, float len, float blur) {
        float L = max(len, 1e-2);
        float s = sqrt(0.8 + 0.25 * blur * blur);
        float gauss = exp(-0.5 * q.y * q.y / (s * s)) / (s * 2.5066) * (normalCDF((L - q.x) / s) - normalCDF(-q.x / s)) / L;
        if (blur < 3) return gauss;
        // The blur disc, averaged over two radii a pixel apart for a soft rim.
        float disc = 0;
        for (int i = 0; i < 2; i++) {
            float r = blur - 0.5 + float(i);
            float h2 = r * r - q.y * q.y;
            if (h2 <= 0) continue;
            float h = sqrt(h2);
            disc += 0.5 * max(min(q.x + h, L) - max(q.x - h, 0.0), 0.0) / (L * PI * r * r);
        }
        return mix(gauss, disc, smoothstep(3.0, 6.0, blur));
    }
    """
}
