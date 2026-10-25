extension SkyWindow {
    /// Metal for the window itself, half a metre from a lens focused on the sky: a painted timber casement with a
    /// glazing bar and a sill, softly out of focus and lit by the daylight coming through it (or a lamp in the room at
    /// night); raindrops on the glass that each hold a tiny upside-down picture of the view; dust and smears that light
    /// up when the Sun shines on the pane; misting low on the glass on cold, damp days; and a faint reflection of the room.
    static let glass = """
    // The window's layout in units of the screen's height, x from the left edge, y from the top.
    struct Pane { float2 p; float aspect; };

    // Signed distance (negative inside the glass) to the panes' edges.
    float paneDistance(float2 p, float aspect, float barX) {
        float side = 0.028, top = 0.026, sill = 0.075, bar = 0.013;
        float2 lo = float2(side, top), hi = float2(aspect - side, 1 - sill);
        float d = max(max(lo.x - p.x, p.x - hi.x), max(lo.y - p.y, p.y - hi.y));
        // The glazing bar splits the window into two panes.
        d = max(d, bar - abs(p.x - barX));
        return d;
    }

    // Raindrops on the outside of the glass: beads that form and dry, and larger drops that run down, leaving trails.
    // Returns the drop's centre (in screen-height units), its normalised offset, and coverage.
    float4 raindrop(float2 p, float time, float amount, thread float2 &centre) {
        float4 best = 0;
        centre = p;
        // Small beads.
        for (int layer = 0; layer < 2; layer++) {
            float scale = layer == 0 ? 38.0 : 17.0;
            float2 g = p * scale;
            float2 cell = floor(g);
            for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++) {
                float2 c = cell + float2(i, j);
                float h = hash12(c + layer * 17.3);
                if (h > amount * (layer == 0 ? 0.3 : 0.14)) continue;
                float life = fract(time * (0.01 + 0.02 * hash12(c + 4.1)) + h * 9);
                float r = (0.18 + 0.22 * hash12(c + 5.0)) * smoothstep(0.0, 0.08, life) * smoothstep(1.0, 0.75, life);
                float2 at = c + 0.5 + (hash22(c + layer) - 0.5) * 0.6;
                float2 d = (g - at) / float2(1.0, 1.12);
                float dist = length(d) / max(r, 1e-3);
                if (dist < 1 && r > best.w) {
                    best = float4(d / max(r, 1e-3), 1 - dist, r);
                    centre = at / scale;
                }
            }
        }
        // Runners: a drop that slides down its column, and the trail of tiny beads it leaves.
        float2 sg = p * float2(7.0, 1.0);
        float col = floor(sg.x);
        float hs = hash11(col * 3.7 + 0.5);
        if (hs < amount * 0.55) {
            float speed = 0.05 + 0.08 * hs;
            float y = fract(time * speed + hs * 10);
            float x = fract(sg.x) - 0.5 + 0.05 * sin(sg.y * 6 + hs * 20) + 0.02 * sin(sg.y * 17 + hs * 7);
            float head = sg.y - y * 1.2 + 0.1;
            float2 d = float2(x * 7.0, head * 22.0);
            float r = 0.55;
            if (length(d) < r) {
                best = float4(d / r, 1 - length(d) / r, 0.5);
                centre = float2((col + 0.5 - (x - (fract(sg.x) - 0.5))) / 7.0, (y * 1.2 - 0.1) / 1.0);
            } else if (head < 0 && head > -0.6 && abs(x) < 0.03) {
                float2 bead = float2(x * 7.0, fract(head * 30) - 0.5);
                float br = 0.22 * (1 + head);
                if (length(bead) < br) {
                    best = float4(bead / br, 1 - length(bead) / br, 0.15);
                    centre = p;
                }
            }
        }
        return best;
    }

    fragment float4 skyGlass(V in [[stage_in]], constant U &u [[buffer(0)]], constant Sky &k [[buffer(1)]],
                             texture2d<float> hdr [[texture(0)]], texture2d<float> blur [[texture(1)]],
                             const device float4 *amb [[buffer(2)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float2 uv = in.uv;
        float aspect = u.res.x / u.res.y;
        float2 p = float2(uv.x * aspect, uv.y);
        float barX = aspect * 0.64;
        float d = paneDistance(p, aspect, barX);
        float E = k.cam.w;
        // The frame is close and the lens is focused far away, so its edges are soft.
        float soft = 0.0035;
        float inGlass = 1 - smoothstep(-soft, soft, d);

        float3 c = hdr.sample(s, uv).rgb;
        // Real overcast and fog are close to neutral grey-white: take the cast out in proportion to the cloud.
        float murk = saturate(max(k.cover.x * 1.1, k.weather.z) + k.weather.x * 0.3);
        float lum = dot(c, float3(0.2126, 0.7152, 0.0722));
        c = mix(c, lum * float3(0.97, 1.0, 1.05), 0.7 * murk * (1 - k.night.x));
        float3 blurred = blur.sample(s, uv).rgb;

        // Light inside the room: by day what the window lets in, at night a warm lamp, both already exposed.
        float3 daylight = amb[0].rgb * E * 0.35 + amb[3].rgb * E * 0.02;
        float3 lamp = k.room.rgb;
        float3 room = daylight * 0.25 + lamp;

        if (inGlass > 0) {
            // Raindrops refract a small inverted image of the view behind them, ringed dark at the rim.
            float rain = k.weather.x;
            if (rain > 0.02) {
                float2 centre;
                float4 drop = raindrop(p, u.time, rain, centre);
                if (drop.z > 0) {
                    float2 cuv = float2(centre.x / aspect, centre.y);
                    float2 look = cuv - (uv - cuv) * 7.0 + float2(0, -0.04);
                    float3 inside = blur.sample(s, clamp(look, 0.0, 1.0)).rgb * 1.1;
                    float rim = smoothstep(0.0, 0.35, drop.z);
                    inside *= 0.35 + 0.65 * rim;
                    // A highlight of the bright sky above, top-left on each drop.
                    float spec = smoothstep(0.55, 0.9, dot(drop.xy, normalize(float2(-0.5, -0.8)))) * smoothstep(0.0, 0.25, drop.z);
                    inside += spec * blur.sample(s, float2(uv.x, 0.15)).rgb * 0.5;
                    c = mix(c, inside, smoothstep(0.0, 0.08, drop.z));
                }
                // A film of water softens the view a little.
                c = mix(c, blurred, rain * 0.25);
            }
            // Misting, thickest low on the pane, where the glass is coldest.
            float mist = k.night.z;
            if (mist > 0.01) {
                float m = mist * smoothstep(0.35, 0.95, uv.y + 0.25 * (fbm(p * 3.0, 3) - 0.5)) * (0.75 + 0.25 * fbm(p * 40, 2));
                c = mix(c, blurred * 0.85 + room * 0.15, saturate(m));
            }
            // Dust specks and old smears on the glass, out of focus, lit up when sunlight falls on the pane.
            float3 sunOnGlass = amb[3].rgb / max(k.sun.y, 0.15) * E * saturate(dot(forward(k), k.sun.xyz)) * k.lightDir.w;
            float2 sunUV = screenOf(k.sun.xyz, k, aspect);
            float near = exp(-length((uv - sunUV) * float2(aspect, 1)) * 2.2);
            float specks = 0;
            float2 g = p * 140;
            float2 cell = floor(g);
            float hcell = hash12(cell);
            if (hcell < 0.06) {
                float2 at = cell + 0.5 + (hash22(cell) - 0.5) * 0.5;
                specks = smoothstep(0.5, 0.15, length(g - at)) * (0.3 + 0.7 * hash12(cell + 3.3));
            }
            float smear = smoothstep(0.55, 0.85, fbm(float2(p.x * 3 + p.y * 1.5, p.y * 9) + 4.2, 4)) * 0.4 + smoothstep(0.6, 0.9, fbm(p * 7 + 1.3, 3)) * 0.3;
            c += sunOnGlass * (specks * 0.004 + smear * 0.0012) * k.extra.y * (0.15 + near);
            // The room, faintly reflected in the glass: about 4% of it, seen mostly where the view is dark.
            float wash = 0.6 + 0.4 * (1 - uv.y);
            float lampSpot = exp(-length((p - float2(aspect * 0.12, 0.92)) * float2(1.0, 1.6)) * 5.0);
            c += (daylight * 0.25 * wash + lamp * (0.25 + 2.5 * lampSpot)) * k.room.w;
        }

        // The timber frame: painted white, lit by the room and by the window light on its inner edges.
        float3 paint = float3(0.78, 0.76, 0.72);
        float grain = 0.94 + 0.06 * fbm(float2(p.x * 2, p.y * 60) + 3.1, 3);
        float bevel = exp(-max(d, 0.0) / 0.006);
        float3 fc = paint * grain * (room * 0.9 + daylight * 0.35 * bevel);
        // The sill's top catches the most light from the window.
        float sillTop = smoothstep(1 - 0.075 - soft, 1 - 0.075 + 0.004, p.y) * (1 - smoothstep(1 - 0.06, 1 - 0.045, p.y));
        fc += paint * daylight * 0.5 * sillTop;
        fc *= 1 - 0.35 * smoothstep(1 - 0.045, 1.0, p.y);
        c = mix(fc, c, inGlass);
        return float4(c, 1);
    }
    """
}
