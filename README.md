# Kontrfaktyki odporne na concept drift (Counterfactuals by Design)

Projekt dotyczy **kontrfaktyków w strumieniach danych z dryfem pojęć**. Kontrfaktyk, który dziś jest poprawny
(zmienia decyzję klasyfikatora), może przestać być poprawny po przesunięciu granicy decyzyjnej. Pytanie:
**czy da się tak uczyć klasyfikator, żeby kontrfaktyki były odporne na przyszły drift?**

Generator kontrfaktyków to **DiCE** (`dice-ml`, wariant gradientowy), używany bez modyfikacji. Różnica między
porównywanymi podejściami leży wyłącznie w **klasyfikatorze** i progu `tau` (min. P(klasa docelowa) wymagane od
kontrfaktyka).

> Status: **ok. 50% pracy** – część podstawowa oraz eksperymenty A, B i C gotowe; rozszerzenia i raport do zrobienia (patrz [Plan dalszych prac](#plan-dalszych-prac)).

## Struktura

```
cfd/
  streams.py    strumienie SEA i Hyperplane, formalnie ograniczony drift, DriftAssumption
  models.py     klasyfikatory: standard, CbD, Dual Decision Boundary
  cf.py         opakowanie DiCE
  metrics.py    odporność czasowa R, bliskość, rzadkość, dokładność pod driftem
  evaluate.py   protokół eksperymentalny (strumień x ziarno x metoda), tabele zbiorcze
  plots.py      wykresy
experiments/
  run_basic.py     część podstawowa
  run_advanced.py  część zaawansowana (eksperymenty A, B, C)
tests/
  test_basic.py    szybkie testy poprawności
results/
  basic/           wyniki części podstawowej
  advanced/        wyniki części zaawansowanej
```

## Uruchomienie

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python tests/test_basic.py                     # testy
python experiments/run_basic.py [--quick]      # część podstawowa  -> results/basic
python experiments/run_advanced.py [--quick] [--only A B C]   # część zaawansowana -> results/advanced
```

## Model driftu i metryka odporności

- Koncept strumienia opisuje skalar `delta`: `y = f(x; phi(delta))`, `phi(0)` = koncept początkowy.
  - **SEA:** klasa 1 <=> `x1 + x2 <= theta0 + delta`
  - **Hyperplane:** `w(delta) = w0 + delta * s`, klasa 1 <=> `w.x >= 0.5 * sum(w)`
- Drift jest **ograniczony**: `|delta| <= Delta`, prędkość zmian `<= v`.
- **Odporność czasowa kontrfaktyka** `R` = zakres przyszłego driftu (w obu kierunkach, najgorszy z nich),
  przy którym kontrfaktyk nadal ma klasę docelową. Raportujemy `R/Delta` oraz odsetek kontrfaktyków
  przeżywających pełne `Delta` (`robust_at_Delta`) i `Delta/2` (`robust_at_half`).
- Pozostałe metryki: bliskość (L1, L2), rzadkość (liczba zmienionych cech), skuteczność DiCE, dokładność
  modelu w `t=0` i uśredniona po driftcie.

## Porównywane metody

| Metoda | Idea |
|---|---|
| **Standard** | MLP uczony na etykietach z `t=0` (baza). |
| **CbD** (Counterfactuals by Design) | Jedna granica. Strata: `(1-alpha)*BCE(f, y) + alpha*BCE(f, q_mean)`, gdzie `q_mean` = ułamek scenariuszy driftu, w których `x` jest klasy 1. Próg `tau` w DiCE wybiera region odporny. |
| **Dual Decision Boundary** | Trzy wyjścia: głowa klasyfikacyjna `f_c` oraz głowy odporności `f_r1`, `f_r0`, uczone na regionach „bezpiecznych" (klasa niezmienna we wszystkich scenariuszach driftu). Kontrfaktyk ma dojść do `f_r{cel} >= tau`, a nie tylko przekroczyć `f_c = 0.5`. |

Obie metody wymagają założenia o przyszłym driftcie (`DriftAssumption`): zakres, opcjonalnie kierunek.

## Wyniki dotychczasowe

### Część podstawowa (`results/basic/`)

Kontrfaktyki DiCE na standardowym klasyfikatorze szybko tracą ważność:

| Strumień | Przeżywa pełne `Delta` | Średnie `R/Delta` |
|---|---|---|
| SEA | **0%** | 0,66 |
| Hyperplane | 56% | 1,21 |

Wykresy: `fig_01` (strumienie i drift), `fig_02` (dokładność zamrożonego modelu w czasie),
`fig_03` (przykład DiCE vs przesuwająca się granica), `fig_04` (zanik poprawności kontrfaktyków).

### Eksperyment A – porównanie główne (`results/advanced/`)

6 wariantów x 2 strumienie x 3 ziarna x 40 zapytań na klasę docelową.

| Strumień | Metoda | `R/Delta` | Przeżywa `Delta` | L1 | Sukces DiCE | Acc (po driftcie) |
|---|---|---|---|---|---|---|
| SEA | Standard (tau=0,5) | 0,66 | 0% | 0,34 | 100% | 0,906 |
| SEA | CbD (tau=0,9) | 1,00 | 57% | 0,42 | 82,5% | 0,908 |
| SEA | **Dual (tau=0,5)** | **1,46** | **100%** | 0,49 | 100% | 0,908 |
| Hyperplane | Standard (tau=0,5) | 1,21 | 56% | 0,55 | 100% | 0,946 |
| Hyperplane | CbD (tau=0,9) | 1,30 | 65% | 0,60 | 100% | 0,949 |
| Hyperplane | **Dual (tau=0,5)** | **1,66** | **91%** | 0,79 | 100% | 0,947 |

Pełna tabela: `results/advanced/A_main_comparison.md`. Wykresy: `fig_A_survival.png`, `fig_A_box.png`.

**Wnioski wstępne**
- Dual Boundary daje najwyższą odporność przy niezmienionej dokładności modelu; koszt to większa odległość
  kontrfaktyka od zapytania (SEA: L1 0,49 vs 0,34).
- CbD przy `tau=0,5` nie poprawia odporności, trzeba podnieść `tau` (kosztem skuteczności DiCE: 82,5% na SEA).
- Podniesienie `tau` w modelu standardowym poprawia odporność tylko częściowo (SEA: 0% -> 18% przeżywających `Delta`).
- Kontrola na czystych etykietach pokazuje, że poprawa nie wynika z samego usunięcia szumu.

### Eksperyment B – kompromis bliskość / odporność (`B_tradeoff.*`, `fig_B_tradeoff.png`)

Przegląd `tau` i `alpha` (2 ziarna x 20 zapytań na klasę docelową, więc wyniki są orientacyjne).

| Strumień | Metoda | `R/Delta` | Przeżywa `Delta` | L1 | Sukces DiCE |
|---|---|---|---|---|---|
| SEA | Standard tau=0,8 | 0,92 | 15% | 0,37 | 100% |
| SEA | CbD a=0,7 tau=0,9 | 1,00 | 57% | 0,40 | 76% |
| SEA | CbD a=0,3 tau=0,9 | 1,16 | 100% | 0,43 | **35%** |
| SEA | Dual (dowolne tau) | 1,41 | 100% | 0,46 | 100% |
| Hyperplane | Standard tau=0,8 | 1,27 | 61% | 0,59 | 100% |
| Hyperplane | CbD a=0,3 tau=0,9 | 1,51 | 84% | 0,79 | 100% |
| Hyperplane | Dual (dowolne tau) | 1,63–1,65 | 91% | 0,79–0,81 | 100% |

- Odporność kosztuje bliskość: im większe `R/Delta`, tym większe L1. Dual leży na krawędzi tego kompromisu.
- CbD może zbliżyć się do odporności Dual tylko przez obniżenie `alpha` i podniesienie `tau`, ale na SEA DiCE
  przestaje wtedy znajdować kontrfaktyki (35% sukcesu). Dual osiąga odporność przy 100% sukcesu.
- Dokładność modeli różni się w trzecim miejscu po przecinku (0,907–0,909 na SEA), więc odporność nie odbywa się kosztem dokładności.
- **Do wyjaśnienia:** wyniki Dual nie zależą od `tau`, a Standard i CbD dają identyczne wyniki dla `tau` 0,5–0,7.
  Wygląda na to, że warunek stopu w DiCE nie jest tam wiążący (kontrfaktyk i tak przekracza próg). Trzeba to sprawdzić.

### Eksperyment C – błędne założenia o driftcie (`C_misspecification.*`, `fig_C_misspecification.png`)

Dual Boundary trenowany z założonym zakresem `Delta_train = b * Delta` (rzeczywiste `Delta` strumienia = 1).

| Strumień | Założenie | `R/Delta` | Przeżywa `Delta` | L1 |
|---|---|---|---|---|
| SEA | Standard | 0,67 | 0% | 0,31 |
| SEA | zakres 0,5 Delta (za mały) | 0,95 | 45% | 0,37 |
| SEA | zakres 1,0 Delta (poprawny) | 1,41 | 100% | 0,46 |
| SEA | zakres 2,0 Delta (za duży) | 2,00 | 100% | 0,66 |
| Hyperplane | Standard | 1,20 | 58% | 0,56 |
| Hyperplane | zakres 0,5 Delta | 1,45 | 75% | 0,78 |
| Hyperplane | zakres 1,0 Delta | 1,63 | 91% | 0,79 |
| Hyperplane | zakres 2,0 Delta | 1,94 | 100% | 0,93 |

- **Za mały zakres** daje odporność tylko do założonej granicy, ale nadal lepszą niż Standard.
- **Za duży zakres** jest bezpieczny, ale kosztuje bliskość (SEA: L1 0,66 vs 0,46). Nadmiarowa ostrożność nie psuje wyniku, tylko go drożej kupuje.
- **Zły kierunek** (założony drift `[0, Delta]` albo `[-Delta, 0]`, rzeczywisty w obu kierunkach): średnia odporność
  spada (SEA: 1,00 i 0,94 vs 1,41), a poprawność kontrfaktyków w modelu spada nawet do 92,5%. W niezałożonym kierunku
  część kontrfaktyków wygasa wcześniej niż w modelu standardowym (prawy panel rysunku). Znajomość samego zakresu
  jest więc bezpieczniejsza niż zakładanie kierunku.

## Plan dalszych prac

Zostało ok. miesiąca. Plan w czterech etapach:

**Etap 1 – wiarygodność wyników**
- [ ] Więcej ziaren (np. 10) zamiast 3, odchylenia standardowe / przedziały ufności w tabelach.
- [ ] Test istotności statystycznej (Wilcoxon) dla Standard vs CbD vs Dual.
- [ ] Wyjaśnić, dlaczego wyniki nie zależą od `tau` w Dual i w Standard/CbD dla `tau` 0,5–0,7 (patrz eksperyment B).
- [ ] Powtórzyć B i C na pełnej próbie (obecnie 2 ziarna x 20 zapytań).
- [ ] Dodatkowa metoda bazowa: kontrfaktyk z marginesem (przesunięcie za granicę o stały margines) – uczciwy
      konkurent, który wymusza odporność bez zmiany modelu.

**Etap 2 – ablacje i uogólnienie**
- [ ] Ablacje Dual/CbD: waga `beta`, liczba scenariuszy `K`, rozmiar sieci.
- [ ] Drugi generator kontrfaktyków (np. Wachter), żeby pokazać, że zysk nie zależy od DiCE.
- [ ] Inne typy driftu: nagły, cykliczny, losowy (random walk) – czy metody działają poza założonym liniowym driftem.

**Etap 3 – realistyczniejszy scenariusz**
- [ ] Symulacja strumienia w czasie: model z okresowym douczaniem, kontrfaktyk generowany w `t` i sprawdzany w `t+k`.
- [ ] Dodatkowy strumień (RBF / Agrawal) lub zbiór rzeczywisty (np. Electricity).
- [ ] Estymacja zakresu driftu z danych (detektor ADWIN/DDM) zamiast zakładania znanego `Delta`.
- [ ] Dodatkowe metryki: plausibilność, różnorodność (wiele kontrfaktyków na zapytanie).

**Etap 4 – dokumentacja**
- [ ] Raport: wstęp, metoda, protokół, wyniki, wnioski, ograniczenia.
- [ ] Prezentacja wyników.

## Ograniczenia (stan obecny)

- Tylko dwa strumienie syntetyczne i jeden typ driftu (liniowy, ograniczony).
- Małe próby: 3 ziarna, 40 zapytań na klasę docelową; brak przedziałów ufności.
- Odporność liczona względem znanej funkcji etykiety strumienia, więc nie da się jej wprost policzyć na zbiorach rzeczywistych.
- CbD i Dual zakładają znajomość zakresu driftu (eksperyment C sprawdza wrażliwość na błąd tego założenia).
- Jeden generator kontrfaktyków (DiCE).
