# TD3 Rocket Flip + Hover Retrospective

Bu dosya, mevcut projede TD3 ile roketi takla attirip tekrar hover'a sokmaya calisirken ogrendigimiz notlari saklamak icin yazildi. Yeni projeye baslarken ayni hatalara dusmemek ve daha kisa yoldan ilerlemek icin referans olarak kullan.

## Proje Amaci

Hedef davranis:

1. Roket yukselir.
2. Belirli irtifada takla atar.
3. Takladan sonra tekrar upright hale gelir.
4. Dususu motorla kirar.
5. Yaklasik 5m civarinda stabil hover yapar.

Bu projede SAC dosyasina dokunmadan TD3 tarafini gelistirmeye calistik. Ortak env degisikliklerini mumkun oldugunca opsiyonel TD3 hook'lari olarak tuttuk.

## Ana Dosyalar

- `hopper_env.py`: MuJoCo environment, fizik, observation, phase ve reward hesaplari.
- `td3.py`: TD3'e ozel reward agirliklari, env kwargs ve action noise.
- `rl_common.py`: train/watch ortak akisi, CSV loglama, progress bar.
- `rl.py`: CLI girisi.
- `hopper_default.xml`: MuJoCo model ve thrust sensorleri.

## En Iyi Hatirlanan Run

En iyi oldugunu dusundugumuz run:

```text
runs/td3_hopper_20260529_143745
```

Bu run basarili hover'a ulasmadi ama diger kotu run'lara gore recovery davranisi daha canliydi.

Ozet:

- `flip`: 241690 step
- `recovery`: 6213 step
- `hover`: 0 step
- `flip_low_altitude_stall`: 205
- `recovery_low_altitude_stall`: 14
- Recovery max height yaklasik `3.49m`
- Recovery thrust ortalamasi yaklasik `4.66N`

Bu da halen iyi degil; fakat recovery'ye girme davranisi sonraki bazi denemelere gore daha iyiydi.

## Kotulestiren Denemeler

### Recovery low-thrust ceza paketi

Sonradan su tarz sinyaller ekledik:

- Dusuk thrust'a recovery'de sert ceza.
- Upright thrust'a ekstra odul.
- Yatik tirmana ceza.

Sonuc kotulesti. Model recovery'ye girmek yerine recovery phase'inden kacmayi ogrendi.

Kotu run ornegi:

```text
runs/td3_hopper_20260529_153702
```

Ozet:

- `recovery`: sadece 256 step
- Recovery thrust mean: yaklasik `0.9N`
- Hover yok

Ders: Ceza cok dogrudan ve sert olursa model hedef davranisi ogrenmek yerine phase'e girmekten kacar.

### Alcap start override hatasi

`td3.py` icindeki `ENV_KWARGS` icine su degerler eklenmisti:

```python
"start_z": 0.8,
"random_start_z": True,
"min_start_z": 0.5,
"max_start_z": 1.2,
```

Bu, watch komutunda `--fixed-start-z --start-z 9.3` verilse bile roketi 0.5-1.2m araligindan baslatiyordu.

Ders: CLI ile verilen start ayarlarini `ENV_KWARGS` icinde override etme. `ENV_KWARGS` phase/reward/fizik hook'lari icin kullanilsin, start araligi CLI'dan gelsin.

## Ise Yarayan Parcalar

### TD3 action noise

TD3 deterministic oldugu icin action noise kritik oldu.

Kullanilan fikir:

```python
NormalActionNoise(
    mean=np.zeros(3),
    sigma=np.array([0.08, 0.25, 0.25]),
)
```

Thrust noise daha dusuk, TVC yaw/pitch noise daha yuksek tutuldu.

### Task-state observation

TD3 icin observation'a phase state eklemek faydaliydi:

- phase one-hot
- flip progress
- hover timer fraction
- phase target z farki

Ders: Ayni fiziksel state farkli phase'lerde farkli hedef anlamina geliyor. Bu bilgi verilmezse TD3 kararsiz policy ogreniyor.

### Flip low-altitude stall

Model recovery cezasindan kacmak icin flip phase'te yerde oyalanmayi ogrendi. Bunu kapatmak icin:

- `flip_progress >= 0.75`
- `z < 2.0m`
- upward speed `< 0.25m/s`
- bu durum `0.8s` surerse fail

Bu kural kacis yolunu kapatti ama tek basina hover ogrenmeye yetmedi.

### Recovery low-altitude stall

Recovery'de yerde motor basip bekleme davranisini kapatmak icin:

- `z < 2.0m`
- upward speed `< 0.25m/s`
- bu durum `1.0s` surerse fail

Ders: Bu kural gerekli ama full task icinde cok erken uygulanirsa model recovery'ye girmekten kacabilir.

## Ana Problem

Full pipeline icinde reward yama yapa yapa su kisir donguye girdik:

1. Flip iyi oluyor.
2. Recovery kotu oldugu icin yere dusuyor.
3. Recovery'yi cezalandirinca model recovery'den kaciyor.
4. Flip'te kacisi kapatinca model yine baska lokal cozum buluyor.

Yani full gorev icinde recovery-hover becerisini sifirdan ogrenmeye calismak verimsiz oldu.

## Yeni Projede Onerilen Yol

Yeni projede tek bir full-task egitimle baslama. Curriculum ile git.

### 1. Hover-only environment

Baslangic:

- z: 4.5-5.5m
- orientation: upright
- dusuk linear/angular velocity

Hedef:

- 5m civarinda stabil hover
- upright kalma
- horizontal drift azaltma
- vertical speed azaltma

Bu asamada flip yok, recovery yok.

### 2. Recovery-hover environment

Baslangic:

- z: 2.5-6.0m
- upright veya yari-upright
- negatif vertical velocity
- az/orta horizontal drift

Hedef:

- dususu motorla kir
- upright kal
- 5m hover bandina cik
- hover timer biriktir

Bu asama ogrenilmeden full flip gorevine gecme.

### 3. Flip-only environment

Baslangic:

- z: 8.5-10.5m
- upright
- hedef: taklayi tamamlamak

Hedef:

- flip progress 1.0 civari
- off-axis rotation az
- cok alcağa inmeden upright'a donmek

### 4. Full pipeline

Sonra asamalari birlestir:

```text
climb -> flip -> recovery -> hover
```

Full pipeline'da rewardlar daha hafif olmali. Asil beceriler once alt gorevlerde ogrenilmis olmali.

## Yeni Projede Kacinilacak Hatalar

- Start ayarlarini env kwargs icinde CLI uzerine override etme.
- Sadece ceza ekleyerek davranis zorlamaya calisma; model phase'den kacabilir.
- Recovery-hover ogrenmeden full flip pipeline'i uzun sure train etme.
- Hover success'i sadece phase'e girdi diye verme; stability gate kullan.
- CSV observation kolonlarini sabit varsayma; env dynamic observation isimlerini kullansin.
- SAC ve TD3 ortak dosyalarinda default davranisi degistirme; TD3 hook'larini default kapali tut.
- Final zip'e guvenme; ara checkpointleri mutlaka karsilastir.

## Faydalı Komutlar

Son model izleme:

```powershell
.\venv\Scripts\python.exe rl.py watch --algo td3 --model runs\td3_hopper_20260529_143745\td3_hopper_latest.zip --fixed-start-z --start-z 9.3
```

Checkpoint izleme:

```powershell
.\venv\Scripts\python.exe rl.py watch --algo td3 --model runs\td3_hopper_20260529_143745\checkpoints\td3_hopper_2075000.zip --fixed-start-z --start-z 9.3
```

Train ornegi:

```powershell
.\venv\Scripts\python.exe rl.py train --algo td3 --resume runs\td3_hopper_20260528_202804\checkpoints\td3_hopper_2050000.zip --timesteps 250000 --chunk-steps 25000 --fixed-start-z --start-z 9.3
```

## Son Karar

Bu projede flip davranisi belirli seviyeye geldi, fakat recovery-hover full task icinde yeterince ogrenilemedi. Yeni projede en buyuk zaman kazanci, recovery-hover becerisini ayri bir curriculum environment olarak ogretmek olacak.

Once hover, sonra recovery-hover, sonra flip, en son full pipeline.
