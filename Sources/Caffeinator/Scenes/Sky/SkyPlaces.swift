import Foundation

/// Named places and seas for describing what a satellite sees: "sunrise over Borneo, night across the western Pacific".
/// Big regions (oceans, continents) are the ones used for "night across" and "day over".
enum SkyPlaces {
    struct Place { let name: String; let lat: Double; let lon: Double; let big: Bool }

    static let all: [Place] = table.split(separator: ";").compactMap { entry in
        let f = entry.split(separator: "|")
        guard f.count == 3, let lat = Double(f[1]), let lon = Double(f[2]) else { return nil }
        let big = f[0].hasPrefix("*")
        return Place(name: String(big ? f[0].dropFirst() : f[0]), lat: lat, lon: lon, big: big)
    }

    /// Great-circle angle between two points, degrees.
    static func angle(_ lat1: Double, _ lon1: Double, _ lat2: Double, _ lon2: Double) -> Double {
        let r = Double.pi / 180
        let c = sin(lat1 * r) * sin(lat2 * r) + cos(lat1 * r) * cos(lat2 * r) * cos((lon1 - lon2) * r)
        return acos(max(-1, min(1, c))) / r
    }

    static func nearest(_ lat: Double, _ lon: Double, big: Bool = false) -> Place? {
        all.filter { !big || $0.big }.min { angle(lat, lon, $0.lat, $0.lon) < angle(lat, lon, $1.lat, $1.lon) }
    }

    private static let table = """
    *the North Pacific|38|-160;*the western Pacific|12|150;*the central Pacific|2|-170;*the eastern Pacific|10|-115;\
    *the South Pacific|-30|-125;*the southwest Pacific|-28|172;*the North Pacific|45|170;the Philippine Sea|20|132;\
    the South China Sea|13|114;the East China Sea|29|126;the Sea of Japan|40|135;the Sea of Okhotsk|53|150;\
    the Bering Sea|58|-178;the Gulf of Alaska|56|-145;the Coral Sea|-16|155;the Tasman Sea|-38|160;the Arafura Sea|-10|134;\
    the Java Sea|-5|111;the Bay of Bengal|14|88;the Arabian Sea|15|63;*the Indian Ocean|-15|80;*the southern Indian Ocean|-38|75;\
    the western Indian Ocean|-8|55;the eastern Indian Ocean|-10|97;*the Southern Ocean|-58|30;*the Southern Ocean|-60|-100;\
    *the Southern Ocean|-60|150;*the Southern Ocean|-56|95;*the North Atlantic|40|-40;*the tropical Atlantic|10|-35;\
    *the South Atlantic|-25|-15;*the South Atlantic|-40|-35;the Caribbean|15|-75;the Gulf of Mexico|25|-90;the Labrador Sea|58|-55;\
    the Norwegian Sea|68|5;the North Sea|56|3;the Mediterranean|36|18;the Black Sea|43|34;the Caspian Sea|42|51;the Red Sea|20|38;\
    the Persian Gulf|27|51;the Gulf of Guinea|2|3;the Mozambique Channel|-18|41;Hudson Bay|60|-85;\
    Japan|36|138;Korea|37|127.5;northern China|40|113;eastern China|31|117;southern China|24|110;western China|38|85;Tibet|32|88;\
    Mongolia|46|103;eastern Siberia|63|135;central Siberia|62|100;western Siberia|60|75;Kazakhstan|48|67;Kamchatka|56|160;\
    Taiwan|23.7|121;the Philippines|12|122;Vietnam|16|107;Thailand|15|101;Myanmar|21|96;Malaysia|3.5|102;Borneo|1|114;\
    Sumatra|0|101;Java|-7.5|110;New Guinea|-5|142;Sulawesi|-2|121;northern Australia|-17|133;Western Australia|-26|121;\
    eastern Australia|-25|147;south-eastern Australia|-36|146;central Australia|-24|134;*Australia|-25|134;New Zealand|-41|174;\
    Hawaii|20.5|-157;Fiji|-17.5|178;French Polynesia|-17|-150;India|22|79;southern India|13|78;northern India|27|80;\
    Sri Lanka|7.8|80.7;Pakistan|29|69;the Himalaya|28.5|84;Bangladesh|24|90;Afghanistan|34|66;Iran|32|54;Arabia|23|46;\
    Turkey|39|35;Iraq|33|44;Egypt|27|30;*the Sahara|23|10;the Sahel|14|5;West Africa|10|-8;Nigeria|9|8;the Congo basin|-1|22;\
    East Africa|-3|36;the Horn of Africa|8|45;Ethiopia|9|39;Sudan|15|30;Libya|27|17;Morocco|32|-6;Angola|-12|18;\
    *southern Africa|-25|25;South Africa|-30|24;Madagascar|-19|46.7;Namibia|-22|17;*Africa|5|20;Spain|40|-4;France|46.5|2.5;\
    Britain and Ireland|54|-3;Italy|43|12.5;Germany|51|10;Scandinavia|63|15;Poland|52|19;Ukraine|49|32;western Russia|56|45;\
    the Balkans|43|21;Greece|39|22;Iceland|65|-18;Greenland|72|-40;*Europe|50|15;*Asia|45|90;*Siberia|62|105;\
    Alaska|64|-152;western Canada|56|-120;central Canada|55|-100;eastern Canada|50|-70;the Pacific Northwest|46|-122;\
    California|37|-120;the Rockies|42|-108;the Great Plains|40|-99;the Midwest|42|-89;the American Northeast|42|-74;\
    the American Southeast|33|-84;Florida|28|-81.5;Texas|31|-99;Mexico|23|-102;Central America|13|-86;Cuba|21.5|-79;\
    Colombia|4|-73;Venezuela|7|-66;the Amazon|-4|-62;Peru|-10|-75;Bolivia|-17|-64;eastern Brazil|-12|-42;southern Brazil|-25|-50;\
    Argentina|-34|-64;Patagonia|-45|-69;Chile|-30|-71;the Galápagos|-0.6|-90.5;*North America|45|-100;*South America|-12|-58;\
    *Antarctica|-80|0;*Antarctica|-76|120;*Antarctica|-76|-100;*the Arctic|82|0
    """
}
