import random
from datetime import datetime
import numpy as np
import pandas as pd
import time

# Name Pools
ADLAR_ERKEK = [
    "Ahmet", "Mehmet", "Ali", "Hüseyin", "Mustafa", "İbrahim", "Osman", "Hakan",
    "Yusuf", "Murat", "Ömer", "Hasan", "Ramazan", "Kemal", "Serkan", "Emre",
    "Can", "Burak", "Gökhan", "Yasin", "Kadir"
]

ADLAR_KADIN = [
    "Ayşe", "Fatma", "Zeynep", "Elif", "Hatice", "Öznur", "Sema", "Dilara",
    "Beyza", "Büşra", "Rabia", "Seda", "Merve", "Gülsüm", "Neslihan", "Ebru",
    "Selin", "Gizem"
]

ADLAR_UNISEX = ["Deniz", "Umut", "Özgür"]
BUTUN_ADLAR = ADLAR_ERKEK + ADLAR_KADIN + ADLAR_UNISEX

SOYADLAR = [
    "Yılmaz", "Demir", "Çelik", "Kaya", "Öztürk", "Şahin", "Aydın", "Yıldız", "Koç", "Arslan",
    "Yalçın", "Polat", "Kurt", "Güçlü", "Tekin", "Aslan", "Taş", "Akın", "Bulut", "Yıldırım",
    "Ersoy", "Korkmaz", "Kaplan", "Ünal", "Aksoy", "Özdemir", "Karaca", "Şener", "Pala", "Köse",
    "Özcan", "Sarı", "Yurt", "Avcı", "Kılıç", "Baştürk", "Çetin", "Yüksel", "Yavuz", "Şentürk"
]

ILLER = [
    "Adana", "Adıyaman", "Afyonkarahisar", "Ağrı", "Amasya", "Ankara", "Antalya", "Artvin", "Aydın", "Balıkesir",
    "Bilecik", "Bingöl", "Bitlis", "Bolu", "Burdur", "Bursa", "Çanakkale", "Çankırı", "Çorum", "Denizli",
    "Diyarbakır", "Edirne", "Elazığ", "Erzincan", "Erzurum", "Eskişehir", "Gaziantep", "Giresun", "Gümüşhane", "Hakkâri",
    "Hatay", "Isparta", "Mersin", "İstanbul", "İzmir", "Kars", "Kastamonu", "Kayseri", "Kırklareli", "Kırşehir",
    "Kocaeli", "Konya", "Kütahya", "Malatya", "Manisa", "Kahramanmaraş", "Mardin", "Muğla", "Muş", "Nevşehir",
    "Niğde", "Ordu", "Rize", "Sakarya", "Samsun", "Siirt", "Sinop", "Sivas", "Tekirdağ", "Tokat",
    "Trabzon", "Tunceli", "Şanlıurfa", "Uşak", "Van", "Yozgat", "Zonguldak", "Aksaray", "Bayburt", "Karaman",
    "Kırıkkale", "Batman", "Şırnak", "Bartın", "Ardahan", "Iğdır", "Yalova", "Karabük", "Kilis", "Osmaniye", "Düzce"
]

KAMPANYALAR = [
    "Eğitim Bursu ve Öğrenci Destekleri",
    "Doğa ve Çevre Koruma Projeleri",
    "Afet ve Acil Durum Yönetimi",
    "Toplumsal Gelişim ve Eğitim Seminerleri",
    "Bilim ve Teknoloji Fonu",
    "Kültür ve Sanat Faaliyetleri",
    "Dezavantajlı Gruplara Sosyal Destek",
    "Sürdürülebilir Yaşam Alanları",
    "Yerel Kalkınma Girişimleri",
    "Gönüllülük Yaygınlaştırma Programı"
]

BAGIS_TURLERI = ["Genel Bağış", "Proje Bazlı Fon", "Düzenli Destek", "Sponsorluk"]
KANALLAR = ["Web", "Mobil Uygulama", "Banka Entegrasyonu", "Şube/Ofis", "Çağrı Merkezi", "Sosyal Medya"]
ODEME_SEKLI = ["Kredi Kartı", "EFT/Havale", "Nakit", "Mobil Ödeme"]
UYELIK_TURU = ["Bireysel", "Kurumsal", "Destekçi / Sponsor"]

PARA_BIRIMI_AGIRLIKLARI = {"TRY": 0.88, "USD": 0.08, "EUR": 0.04}
KANAL_AGIRLIKLARI = [0.45, 0.25, 0.15, 0.05, 0.05, 0.05]
PB_KEYS = np.array(list(PARA_BIRIMI_AGIRLIKLARI.keys()))
PB_PROBS = np.array(list(PARA_BIRIMI_AGIRLIKLARI.values()))

# Cinsiyet mapping sözlüğü (set arama yerine)
ERKEK_ADLARI_SET = set(ADLAR_ERKEK)
KADIN_ADLARI_SET = set(ADLAR_KADIN)


def _random_phones_vectorized(n: int) -> np.ndarray:
    operatorler = ["530", "531", "532", "533", "535", "541", "542", "543", "544", "545", "505", "506", "507", "552"]
    prefixler = np.random.choice(operatorler, size=n)
    suffixler = np.random.randint(0, 10, size=(n, 7))
    suffixler_str = np.array([''.join(map(str, s)) for s in suffixler])
    return np.char.add(prefixler, suffixler_str)


def _get_cinsiyet(ad: str) -> str:
    """Hızlı cinsiyet tayini"""
    if ad in ERKEK_ADLARI_SET:
        return "Erkek"
    elif ad in KADIN_ADLARI_SET:
        return "Kadın"
    else:
        return np.random.choice(["Erkek", "Kadın"])


def generate_demo_data(n_bagisci: int = 100, n_bagis: int = 1000, random_seed: int = 1881):
    np.random.seed(random_seed)
    random.seed(random_seed)


    ad_indices = np.random.choice(len(BUTUN_ADLAR), size=n_bagisci, replace=True)
    soyad_indices = np.random.choice(len(SOYADLAR), size=n_bagisci, replace=True)
    
    attempts = 0
    while len(set(zip(ad_indices, soyad_indices))) < n_bagisci and attempts < 5:
        duplike_count = n_bagisci - len(set(zip(ad_indices, soyad_indices)))
        ad_indices[np.random.choice(n_bagisci, size=duplike_count, replace=False)] = np.random.choice(len(BUTUN_ADLAR), size=duplike_count)
        soyad_indices[np.random.choice(n_bagisci, size=duplike_count, replace=False)] = np.random.choice(len(SOYADLAR), size=duplike_count)
        attempts += 1
    
    adlar = np.array(BUTUN_ADLAR)[ad_indices]
    soyadlar = np.array(SOYADLAR)[soyad_indices]
    ad_soyadlar = np.array([f"{ad} {soyad}" for ad, soyad in zip(adlar, soyadlar)])
    
    telefonlar = _random_phones_vectorized(n_bagisci)
    
    cinsiyetler = np.array([_get_cinsiyet(ad) for ad in adlar])
    
    iller = np.random.choice(ILLER, size=n_bagisci)
    uyelik_turleri = np.random.choice(UYELIK_TURU, size=n_bagisci)
    
    df_bagiscilar = pd.DataFrame({
        "No": np.arange(1, n_bagisci + 1),
        "Ad Soyad": ad_soyadlar,
        "Cep Telefon": telefonlar,
        "Üyelik Türü": uyelik_turleri,
        "Temsilci": "Sistem Tanımlı",
        "Cinsiyet": cinsiyetler,
        "Eposta": [f"bagisci{i}@ornekmail.org" for i in range(1, n_bagisci + 1)],
        "Il": iller,
        "Ulke": "Türkiye",
    })

    donor_weights = np.random.exponential(scale=1.0, size=n_bagisci)
    donor_weights /= donor_weights.sum()

    baslangic = datetime(2022, 1, 1)
    bitis = datetime.now()
    toplam_saniye = int((bitis - baslangic).total_seconds())

    bagisci_nos = np.random.choice(df_bagiscilar["No"].values, size=n_bagis, p=donor_weights)
    
    rastgele_saatler = np.random.randint(0, toplam_saniye, size=n_bagis)
    tarihler = pd.to_datetime(baslangic) + pd.to_timedelta(rastgele_saatler, unit='s')
    tarih_strs = tarihler.strftime("%Y-%m-%d %H:%M:%S").values
    
    tutarlar = np.random.lognormal(mean=5.5, sigma=1.0, size=n_bagis)
    tutarlar = np.clip(tutarlar, 30.0, 100000.0)
    tutarlar = np.round(tutarlar, 2)
    
    para_birimleri = np.random.choice(PB_KEYS, size=n_bagis, p=PB_PROBS)
    
    kanal_indices = np.random.choice(len(KANALLAR), size=n_bagis, p=KANAL_AGIRLIKLARI)
    kanallar = np.array(KANALLAR)[kanal_indices]
    
    bagis_turleri = np.random.choice(BAGIS_TURLERI, size=n_bagis)
    kampanyalar = np.random.choice(KAMPANYALAR, size=n_bagis)
    odeme_sekilleri = np.random.choice(ODEME_SEKLI, size=n_bagis)
    
    donor_map = dict(zip(df_bagiscilar["No"], df_bagiscilar["Ad Soyad"]))
    telefonlar_map = dict(zip(df_bagiscilar["No"], df_bagiscilar["Cep Telefon"]))
    iller_map = dict(zip(df_bagiscilar["No"], df_bagiscilar["Il"]))
    
    df_bagislar = pd.DataFrame({
        "Bagis No": np.arange(1, n_bagis + 1),
        "Bagisci No": bagisci_nos,
        "Ad Soyad": np.array([donor_map[no] for no in bagisci_nos]),
        "Cep Telefon": np.array([telefonlar_map[no] for no in bagisci_nos]),
        "Tür": bagis_turleri,
        "Tutar": tutarlar,
        "Para Birimi": para_birimleri,
        "Bagis Tarihi": tarih_strs,
        "Bagis Kanali": kanallar,
        "Kampanya Adi": kampanyalar,
        "Odeme Sekli": odeme_sekilleri,
        "Personel": "Sistem",
        "Adet": 1,
        "Toplam Tutar": tutarlar,
        "Hedef Bölge": "Yurtiçi",
        "Il": np.array([iller_map[no] for no in bagisci_nos]),
        "Ulke": "Türkiye",
    })

    df_bagislar = df_bagislar.sort_values(by="Bagis Tarihi").reset_index(drop=True)
    df_bagislar["Bagis No"] = df_bagislar.index + 1

    return df_bagislar, df_bagiscilar