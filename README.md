# Mehrere Ziele unter Risiko: Skalierungskonstanten, Form und Kopplung – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-attribute-risiko-demo.streamlit.app/)** -->

Eine von elf Demos zur **Entscheidungsanalyse** für die Website „Sebastian Hanisch – Operations Research und Machine Learning“; die anderen zehn stehen unten bei den Nachbarn. Sie hebt die Annahme **„ein Ziel“** auf: jeder Ausgang hat hier **zwei Zahlen** (Gewinn in GE und Servicegrad in %), und ein Einzelnutzen genügt nicht mehr. Man braucht je Ziel einen Einzelnutzen $u_i$ und **Skalierungskonstanten $k_i$**, und die Summe $\Sigma k$ ist keine Formalität.

**Die Auszahlungen, die Servicegrade und die $k_i$ sind ausgedacht. Der Kapazitätsfall ist ein Modell, kein Fall, und keine der Zahlen ist an Menschen gemessen.** Die Häufigkeiten stammen aus Zufallsproblemen (Matrizen bzw. Probleme aus festen Generatoren), nicht aus Fällen.

## Warum dieses Problem

Wer zwei Ziele gegeneinander abwägt, rechnet oft mit einer gewichteten Summe. Die multiplikative Form ist genau dann diese gewichtete Summe, wenn $\Sigma k = 1$ ist; sonst steckt in $c = 1 - \Sigma k$ eine Wechselwirkung (Kreuzterm), und die multiplikative Form hängt dann nicht nur von den **Rändern** jedes Ziels ab, sondern von ihrer **Kopplung** (wie die Ausgänge gemeinsam verteilt sind). Die additive Form kennt die Kopplung nicht: sie ist eine Summe von Einzelnutzen, ihr Erwartungswert hängt nur von den Rändern ab (gemessen: B9). Die Demo beantwortet drei Fragen: **(1) Warum genügt ein Nutzen nicht mehr, und was sagt $\Sigma k$? (2) Wann entscheidet die Form, wann die Kopplung? (3) Wie riskant ist es, Mittelwerte einzusetzen statt den Erwartungsnutzen zu rechnen?**

Im Kapazitätsfall (G: groß bauen, K: klein bauen oder mieten; Spanne des Gewinns 300 GE; zweites Ziel Servicegrad) wählt die additive Form bei $k = (2/5;\ 3/10)$ und linearem Gewinn für jede Kopplung $q$ die Alternative G, die multiplikative nur für $q < q^* = 173/450$ (Befund B3). Der Gewinn allein kippt bei der Risikotoleranz $\rho^* = 555{,}8$ GE; mit dem zweiten Ziel kippt er additiv bei 1268,0 GE und multiplikativ bei 410,9 GE, 1174,0 GE oder gar nicht, je nach Kopplung (B4). In 5000 Zufallsmatrizen mit sicheren Ausgängen ist Platz 1 bei $\Sigma k = 1$ in 0 Fällen verschieden, bei $\Sigma k = 0{,}5$ in 448 (B5). Bei gleichen Rändern und anderer Kopplung wechselt Platz 1 nur multiplikativ (B9), und wer die Attribut-Erwartungswerte einsetzt, ändert Platz 1 je nach Krümmung in 2,4 % bis 48,0 % der Probleme (ρ/Spanne 4 bis 0,1; additive Form, 2 Szenarien; B8).

Alles ist **Lehrbuchstoff** (Produktform, Skalierungskonstante K, Skalierungskonstanten als Nutzen der Ecken). Eine Neuheitsaussage gibt es nicht; eigen sind die Vorführung mit Zähleinheit, die Kopplungsmessung und die Rechenfalle bei $\Sigma k$ nahe 1. Alle Formeln und Befunde sind **eigene Nachrechnung in diesem Modell, bei diesen Parametern** (Quellen unten).

## Modell

**Konvention (vor der ersten Gleichung).** Einzelnutzen $u_i(x_i) \in [0, 1]$, normiert auf die Spanne des Ziels (schlechtester Wert 0, bester 1). Beim Gewinn (Spanne −100 bis 200 GE) exponentiell mit Risikotoleranz $\rho$ in GE ($\rho = \infty$: linear), beim Servicegrad linear. **Skalierungskonstanten $k_i$** (nicht „Gewichte“, nicht „Wichtigkeit“): $k_i = U$(Ziel $i$ am besten, alle anderen am schlechtesten). Die Erhebung per Lotterie (sicher: Ecke gegen „mit $p$ alles am besten, sonst alles am schlechtesten“, Indifferenz bei $p = k_i$) ist hier nur Konvention; die $k_i$ sind gesetzt.

**Formen.** Multiplikativ (Produktform): $1 + K\,U = \prod_i (1 + K\,k_i\,u_i)$ mit $1 + K = \prod_i (1 + K\,k_i)$; $K = 0$ (additiv) genau bei $\Sigma k = 1$. Für zwei Ziele ausmultipliziert $U = k_1 u_1 + k_2 u_2 + c\,u_1 u_2$ mit $c = 1 - k_1 - k_2 = K\,k_1 k_2$; $\Sigma k < 1$ heißt $c > 0$, $\Sigma k > 1$ heißt $c < 0$. Additiv: $U = \sum_i (k_i / \Sigma k)\, u_i$.

**Erwartungsnutzen.** $\mathrm{EU} = \sum_z p_z\, U(u(z))$ über die Zellen $z$ der gemeinsamen Verteilung; für zwei Ziele $\mathrm{EU} = k_1 \bar u_1 + k_2 \bar u_2 + c\,(\bar u_1 \bar u_2 + \mathrm{Cov}(u_1, u_2))$. Die additive Form hängt nur von den Rändern ab, der Kovarianzterm gehört der multiplikativen Form allein.

**Kopplung** (eigene Bezeichnung, ein Fachbegriff ist hier nicht verwendet) = gemeinsame Verteilung der Ausgänge bei **gleichen Rändern**: gleichläufig (gute Werte treten zusammen auf), gegenläufig (gut mit schlecht) oder unabhängig.

**Kapazitätsfall (ausgedacht).** Zustände: Nachfrage hoch oder niedrig (je ½), Lieferlage normal oder Engpass (je ½); $q = P(\text{hoch und Engpass})$, $0 \le q \le \tfrac12$ ($q = \tfrac14$ unabhängig, $q = 0$: Engpass nur bei niedriger Nachfrage, $q = \tfrac12$: hohe Nachfrage immer mit Engpass). G: Gewinn 200 GE bei hoher, −100 GE bei niedriger Nachfrage; Servicegrad 100 % bei normaler Lage, 50 % im Engpass. K: Gewinn 30 GE sicher; Servicegrad 100 % bzw. 60 % im Engpass. Voreinstellung $k = (2/5;\ 3/10)$.

**Zufallsprobleme (keine Fälle).** Gewichte $w_i \propto 1/(i+1)$ (Summe 1; für zwei Ziele 0,6 und 0,4). *Form (sichere Ausgänge):* $n$ Alternativen, $m$ Ziele, Einzelnutzen linear und gleichverteilt, additiv mit $w_i$, multiplikativ mit $k_i = \Sigma k \cdot w_i$. *Risiko:* 10 Alternativen, 3 Ziele, $S$ gleich wahrscheinliche Szenarien, Werte gleichverteilt und unabhängig; verglichen werden der Erwartungsnutzen mit gekrümmtem Einzelnutzen ($\rho$ geteilt durch die Spanne) und dieselbe Form auf den Attribut-Erwartungswerten (linear). *Kopplung:* 10 Alternativen, zwei Ziele, je Alternative und Ziel ein Zwei-Punkt-Los (Werte gleichverteilt, je ½), $k = (0{,}6\,\Sigma k;\ 0{,}4\,\Sigma k)$.

**Gilt** für diese Generatoren, den ausgedachten Kapazitätsfall und linearen Einzelnutzen beim Servicegrad. Die Formeln stehen im Expander 📐 der App.

## Methodik

| Einheit | Wie sie arbeitet | Grenze |
|---|---|---|
| **K** (`atr_utility.mult_constant`) | Wurzelsuche (`scipy.optimize.brentq`) für $(e_1 - 1) + e_2 K + e_3 K^2 + \dots = 0$ mit den elementarsymmetrischen Funktionen $e_j$ der $k_i$; $K = 0$ bei $\lvert\Sigma k - 1\rvert < 10^{-12}$ | nötig: $0 \le k_i < 1$, bei $\Sigma k \ne 1$ mindestens zwei positive $k_i$ |
| **Nutzen** (`mult_utility`, `add_utility`) | multiplikativ als `expm1(Σ log1p(K·k_i·u_i))/K` (stabil bei kleinem $K$), additiv als normierte Summe | die direkte Gleichung $(\prod (1 + K k_i) - 1 - K)/K = 0$ löscht sich bei $\Sigma k$ nahe 1 aus (B6) |
| **Kapazitätsfall** (`atr_scenario`) | Erwartungsnutzen über die vier Zellen aufgezählt; Eingaben als `Fraction` rechnen exakt (nur bei $\rho = \infty$), als `float` im Gleitkomma; Kipp-$\rho$ per Raster (17 Punkte 5 GE bis 10⁹ GE) und `brentq`; Kipp-$q$ aus zwei Stützstellen (EU(G) ist in $q$ linear) | „kein Wechsel im Raster“ heißt nicht „für jedes $\rho$“ |
| **Zufallsprobleme** (`atr_sweeps`) | vektorisiert; Platz 1 aus `argmax`, Abstand der beiden besten Werte unter $10^{-12}$ zählt als Gleichstand, nie als Wechsel; Kopplung durch Aufzählung der vier Zellen (hoch/tief × hoch/tief) | Zwei Ziele und Zwei-Punkt-Lose; mehr Ziele sind nicht gemessen |
| **Zufall** (`atr_rng`) | ganzzahliger SplitMix64 (Zähler plus Mischfunktion, vektorisiert), nur gleichverteilte Zahlen (`random()` mit 53 Bit, reine Ganzzahlarithmetik); je Experiment ein eigener Strom aus Zufalls-Seed und Salz | ein Zufalls-Seed ist ein Beispiel; Aussagen stehen als Band über fünf Zufalls-Seeds |
| **Kendall-Tau** (`kendall_tau`) | über alle geordneten Paare ($\tau_a$, ohne Gleichstände), gerechnet an den ersten höchstens 1000 Matrizen | $n < 2$: nicht definiert (NaN) |

Die Live-Rechnung der App braucht je Experiment etwa 0,3 bis 0,6 Sekunden am Reglermaximum (5000 Zufallsprobleme, $n = 40$, $m = 8$, $S = 10$; die Form-Tabelle etwa 0,3 bis 0,4, die Risiko-Tabelle mit multiplikativer Form etwa 0,5, die Kopplung etwa 0,55 Sekunden; Messung auf dem Entwicklungsrechner, nicht festgeschrieben, das Zeichnen kommt hinzu). **Nichts ist vorgerechnet oder eingefroren**; alle Zahlen dieser README rechnen die Tests in `tests/test_claims.py` neu.

## Befunde (gemessen, keine Behauptungen)

Zähleinheit: Matrizen (Form, sichere Ausgänge) bzw. Probleme (Risiko, Kopplung); Standardfehler eines Anteils bei 5000 Einheiten höchstens 0,7 Prozentpunkte. Voreinstellung: **Zufalls-Seed 1017**, 5000 Einheiten; „Band über fünf Zufalls-Seeds“ meint die Zufalls-Seeds 1017, 1, 2, 3, 4. Die Spalte „Test“ nennt die Testfunktion in `tests/test_claims.py` (bzw. die Datei), die genau diese Form prüft. Zählungen aus Neu-Rechnung gelten als Band (± 2), Brüche exakt.

| Nr | Frage | Befund | Test |
|---|---|---|---|
| B1 | Handbeispiel Skalierungskonstante (von Hand, Brüche) | $k = (2/5;\ 3/10)$: $K = (1 - 7/10)/(3/25) = $ **5/2**, die Faktoren sind 2 und 7/4 (2 · 7/4 = 7/2); Kreuzkoeffizient $c = 3/10$. Ecken $U(1,1) = 1$, $U(0,0) = 0$, $U(1,0) = 2/5$, $U(0,1) = 3/10$ (die $k_i$ sind Nutzen der Ecken). Mittelpunkt $(\tfrac12, \tfrac12)$: multiplikativ **17/40**, additiv mit den unnormierten $k_i$ 7/20. Gleiche $k_i$ ($\Sigma k = 7/10$), $A = (1; 0)$, $B = (\tfrac12; \tfrac12)$: additiv A (4/7 gegen 1/2), multiplikativ B (2/5 gegen 17/40); $k = (7/10;\ 3/5)$, $K = -5/7$, $B = (3/5; 3/5)$: additiv B (7/13 gegen 3/5), multiplikativ A (7/10 gegen 84/125). | `test_b1_handbeispiel` |
| B2 | Summe 1 | $\Sigma k = 1$: multiplikative Form gleich additive Form, exakt in Brüchen (4 Werte von $k_1$, je 16 Wertepaare); drei Ziele $k = (0{,}3;\ 0{,}25;\ 0{,}2)$: $K = 1{,}2289$ (Wurzel des Polynoms, `numpy.roots`, 1e-9). | `test_b2_summe_eins_und_drei_ziele` |
| B3 | Kapazitätsfall von Hand (Brüche, $\rho = \infty$, $k = (2/5;\ 3/10)$) | $u_1(30) = 13/30$. EU(K) = **194/375** für jedes $q$; multiplikativ EU(G) = **23/40 − 3q/20**; additiv EU(G) = 17/28, EU(K) = 62/105, **G − K = 1/60 für jedes $q$**. Multiplikativ G − K bei $q = 0$ / ¼ / ½: +0,0577 / +0,0202 / −0,0173. Kipp-Kopplung **$q^* = 173/450$ (0,3844)**; Kovarianz der Einzelnutzen von G: $\tfrac18 - \tfrac{q}{2}$ (null bei $q = \tfrac14$). Die additive Form wählt für jedes $q$ die Alternative G, die multiplikative nur für $q < q^*$. | `test_b3_kapazitaetsfall_von_hand` |
| B4 | Kipp-Risikotoleranz | Gewinn allein: $\rho^* = $ **555,8 GE** (555,801; zwei Rechenwege: aus $u_1(30;\rho) = \tfrac12$ und aus dem Sicherheitsäquivalent 30 GE, 1e-6). Mit Servicegrad (G 50 %, K 60 % im Engpass, $k = (0{,}4;\ 0{,}3)$): **additiv 1268,0 GE** (für jedes $q$), **multiplikativ 410,9 GE ($q = 0$), 1174,0 GE ($q = \tfrac14$), kein Wechsel im Raster ($q = \tfrac12$)** (Raster 5 GE bis 10⁹ GE, 17 Punkte). Bei $\rho = 555{,}8$ GE ist G − K additiv −0,0214, multiplikativ +0,0150 / −0,0225 / −0,0600; Kipp-Kopplung dort $q^* = 0{,}1000$, bei $\rho = 200$ GE gibt es keine. Das zweite Ziel verschiebt $\rho^*$ also um den Faktor 0,74 bis 2,28 gegenüber 555,8 GE. | `test_b4_kipp_risikotoleranz` |
| B5 | Form bei sicheren Ausgängen (20 Alternativen, 5 Ziele, 5000 Matrizen, Zufalls-Seed 1017) | Platz 1 verschieden in **448 / 177 / 0 / 166 / 451** von 5000 Matrizen (9,0 / 3,5 / 0,0 / 3,3 / 9,0 %) bei $\Sigma k = 0{,}5$ / 0,8 / 1,0 / 1,2 / 1,5 ($K = +3{,}72$ / +0,73 / 0 / −0,40 / −0,70). Band über fünf Zufalls-Seeds: $\Sigma k = 0{,}5$: 440 bis 469 (8,8 bis 9,4 %); $\Sigma k = 1{,}5$: 441 bis 481 (8,8 bis 9,6 %); $\Sigma k = 1$: 0 in allen fünf. Die Bänder gelten für diese fünf Zufalls-Seeds und sind schmaler als die Streuung zwischen Zufallsströmen (Standardabweichung eines Zählers rund 20 von 5000); eine Nachprüfung mit einem anderen Generator gab 439 bis 491 ($\Sigma k = 0{,}5$) und 440 bis 483 ($\Sigma k = 1{,}5$). In 4552 von 5000 Matrizen ist die additive Näherung bei $\Sigma k = 0{,}5$ für Platz 1 unauffällig. Rangfolge insgesamt (Kendall-Tau, erste 1000 Matrizen): Mittel 0,950 und kleinster Wert 0,842 ($\Sigma k = 0{,}5$), 0,960 und 0,863 ($\Sigma k = 1{,}5$). | `test_b5_form_sichere_ausgaenge` |
| B6 | Rechenfalle bei $\Sigma k$ nahe 1 (20 · 5, 5000 Matrizen, Zufalls-Seed 1017, $\Sigma k = 0{,}999999$) | $K$ aus den Koeffizienten der Produktentwicklung 2,610e-06 (Näherung $(1 - \Sigma k)/e_2$: 2,610e-06), 0 von 5000 Matrizen mit anderem Platz 1; die direkt gelöste Gleichung gibt $K = $ 1,000e-12 und scheinbar **2 von 5000** (kein Befund, ein Rechenfehler). | `test_b6_rechenfalle` |
| B7 | Form in Abhängigkeit von $m$ und $n$ (3000 Matrizen, Zufalls-Seed 1017; Zähler bei $\Sigma k = 0{,}5$ / 1,5) | $m = 2$: $n = 3$: 209 / 254, $n = 10$: 304 / 680, $n = 40$: 292 / 882; $m = 5$: 121 / 106, 220 / 217, 300 / 335; $m = 8$: 111 / 77, 189 / 178, 257 / 233. Der Anteil steigt von $n = 3$ auf $n = 10$ in allen sechs Zeilen ($m$, $\Sigma k$); von $n = 10$ auf $n = 40$ steigt er in fünf der sechs Zeilen, bei $m = 2$, $\Sigma k = 0{,}5$ (209, 304, 292) bleibt er gleich: 304 gegen 292 von 3000 liegen innerhalb des Standardfehlers der Differenz (etwa 23), ein Rückgang ist nicht belegt. Ursache nicht untersucht. | `test_b7_form_nach_m_und_n` |
| B8 | Mittelwerte einsetzen (10 · 3, 2 Szenarien, 5000 Probleme, Zufalls-Seed 1017, additiv) | Platz 1 ändert sich in **121 / 247 / 495 / 925 / 1628 / 2399** von 5000 Problemen (2,4 / 4,9 / 9,9 / 18,5 / 32,6 / 48,0 %) bei $\rho$/Spanne 4 / 2 / 1 / 0,5 / 0,25 / 0,1. Band über fünf Zufalls-Seeds: $\rho$/Spanne = 2: 238 bis 272 (4,8 bis 5,4 %); 0,1: 2366 bis 2442 (47,3 bis 48,8 %). Linearer Einzelnutzen additiv: 0 von 5000. Der Kapazitätsfall liegt bei $\rho$/Spanne = 555,8/300 = 1,85. Multiplikativ ($\Sigma k = 0{,}5$): 398 / 448 / 574 / 954 / 1608 / 2307 von 5000, **bereits bei linearem Einzelnutzen 392 von 5000** (die Kopplung der Szenarien: Erwartung eines Produkts gegen Produkt der Erwartungen). Szenarienzahl (3000 Probleme, additiv, $\rho$/Spanne = 0,5): $S = 2$ / 3 / 5 / 10: 552 / 600 / 608 / 598 (18,4 bis 20,3 %); bei $\rho$/Spanne = 2: 143 / 159 / 142 / 176. $S = 10$ liegt bei $\rho$/Spanne = 0,5 in allen fünf Zufalls-Seeds über $S = 2$: der Anteil sinkt mit mehr Szenarien **nicht**. | `test_b8_mittelwerte_einsetzen` |
| B9 | Gleiche Ränder, andere Kopplung (10 Alternativen, 2 Ziele, 5000 Probleme, Zufalls-Seed 1017) | **Additiv ändert sich Platz 1 mit der Kopplung in 0 von 5000 Problemen** (alle $\Sigma k$) und bei $\Sigma k = 1$ auch multiplikativ in 0. Multiplikativ, gleichläufig gegen gegenläufig: **570 / 255 / 0 / 270 / 650** von 5000 (11,4 / 5,1 / 0,0 / 5,4 / 13,0 %) bei $\Sigma k = 0{,}5$ / 0,8 / 1,0 / 1,2 / 1,5; gleichläufig gegen unabhängig 327 / 125 / 0 / 130 / 289; gegenläufig gegen unabhängig 258 / 134 / 0 / 144 / 370. Band über fünf Zufalls-Seeds (gleichläufig gegen gegenläufig): $\Sigma k = 0{,}5$: 506 bis 586 (10,1 bis 11,7 %); $\Sigma k = 1{,}5$: 595 bis 701 (11,9 bis 14,0 %). Das Vorzeichen von $c$ entscheidet, welche Kopplung vorn liegt: bei $c > 0$ ist der Erwartungsnutzen gleichläufig nie kleiner als gegenläufig (Mittel +0,0277, größter Unterschied +0,2254 bei $\Sigma k = 0{,}5$), bei $c < 0$ nie größer (in allen 50 000 Alternativen). Unterschied exakt $c \cdot \tfrac12 \cdot \Delta a \cdot \Delta b$ (200 Probleme in Brüchen, `tests/test_oracle_attribute_risiko.py`). | `test_b9_kopplung` |

### Hinweise zu den Zahlen

- **Exakt oder Band.** Brüche (B1 bis B3, die Kopplungs-Identitäten) sind exakt. Die Kipp-Werte (B4) sind Gleitkomma mit relativer Toleranz 1e-6; Zählungen aus Neu-Rechnung (B5 bis B9) gelten als Band ± 2, weil `expm1`, `log1p` und die Summenreihenfolge unter Windows und Linux in der letzten Binärstelle abweichen können. Die Gleichstandsgrenze ist $10^{-12}$.
- **Zufalls-Seed.** Die Zahlen mit „Zufalls-Seed 1017“ sind ein Beispiel; die Bänder über fünf Zufalls-Seeds zeigen, wie weit andere Zufalls-Seeds abweichen. Mit mehr Matrizen (gleicher Zufalls-Seed) bleiben die ersten Matrizen dieselben (Test `tests/test_sweeps.py`).
- **Gewichte.** Alle Zufallsexperimente nutzen $w_i \propto 1/(i+1)$; mit anderen Gewichten (etwa gleichen) ändern sich die Zahlen.
- **Rundung.** Anteile sind einmal gerundet; „9,0 %“ bei 448 von 5000 ist 8,96 %.
- **Gleichstände.** In keinem der Läufe dieser Tabelle trat ein Gleichstand der beiden besten Werte auf (Spalte „Gleichstände“ in der App: 0).

## Befunde und Korrekturen gegenüber der früheren Messung

Eine frühere Messung (Matrizen und Probleme mit einem anderen Zufallsgenerator und mit anderen Gewichten; ihre Zahlen stehen nicht in diesem Repo) wurde hier mit dem SplitMix64 dieser Demo neu gerechnet; die Zahlen dieser README gelten, wo sie abweichen.

1. **Gleiche Befunde, andere Zahlen.** Bei $\Sigma k = 0{,}5$ / 1,5 war Platz 1 früher in 8,9 bis 9,1 % / 8,2 bis 8,9 % der Matrizen verschieden (zwei Zufallsströme, Gewichte (0,35; 0,25; 0,2; 0,1; 0,1)); hier mit $w_i \propto 1/(i+1)$ in 8,8 bis 9,4 % / 8,8 bis 9,6 % (fünf Zufalls-Seeds, B5). Die Gewichte sind eine Regel statt einer freien Liste, damit sie für jedes $m$ definiert sind.
2. **Mittelwerte einsetzen: andere Referenz.** Früher stand die Mittelwert-Rechnung für beide Formen in der additiven Form (multiplikativ mit Risiko: 25,6 % bei $\rho$/Spanne = 0,5, $\Sigma k = 0{,}5$). Hier nutzt die Referenz dieselbe Form wie der Erwartungsnutzen (multiplikativ: 954 von 5000, 19,1 %). Neu (B8): in der multiplikativen Form misst der Anteil Krümmung **und** die Kopplung der Szenarien zusammen: schon bei linearem Einzelnutzen ändert sich Platz 1 in 392 von 5000 Problemen.
3. **Abhängigkeit von $n$.** Die frühere Messung (gleiche Gewichte) fand „keine einfache Richtung in $n$“; mit den harmonischen Gewichten steigt der Anteil von $n = 3$ auf $n = 10$ in allen sechs Zeilen und von $n = 10$ auf $n = 40$ in fünf (B7); in der sechsten ($m = 2$, $\Sigma k = 0{,}5$) bleibt er innerhalb des Standardfehlers gleich.
4. **Kopplung.** Früher 9,9 bis 10,5 % und 12,5 bis 13,1 % (gleichläufig gegen gegenläufig, zwei Zufallsströme); hier 10,1 bis 11,7 % und 11,9 bis 14,0 % (fünf Zufalls-Seeds, B9): dieselbe Größenordnung, die Bänder sind breiter.
5. **Szenarienzahl.** Die Vorannahme „mit mehr Szenarien sinkt der Anteil“ bestätigt sich wie früher nicht (B8); die Ursache ist nicht untersucht.
6. **Kipp-Werte** 555,8 / 1268,0 / 410,9 / 1174,0 GE und $q^* = 173/450$ reproduzieren sich.
7. **Nichts gestrichen.** Alle vier Experimente sind gebaut; die Seite ist lang, aber nicht unhandlich.
8. **Nicht neu gelesen:** die READMEs der Mehrkriterien-Demos `smaa-demo`, `gewichtete-summe-demo`, `gewichte-erheben-demo`, `ordinale-regression-demo`, `mcda-vergleich-demo` und `pareto-dominanz-demo` sind nicht gelesen (kein Abruf möglich); die Abgrenzung unten gilt für das, was gelesen ist.

## Quellen

Produktform, Skalierungskonstante $K$ und die Skalierungskonstanten als Nutzen der Ecken sind Lehrbuchstoff der Mehrattribut-Nutzentheorie. Gelesen (Stufe W, Gliederung und Einleitung): Bell, Keeney, Raiffa (Hrsg.), *Conflicting Objectives in Decisions*, Wiley/IIASA 1977, Kap. 7 (Fishburn): unter Nutzenunabhängigkeit und verallgemeinerter Nutzenunabhängigkeit führt der Weg zur additiven, multiplikativen oder quasi-additiven Form (Einleitung, S. 172; Abschn. 7.3 und 7.4). Eine **Gegenstimme** im selben Band (S. 255 f., sinngemäß): die additive Näherung arbeite bei bedingter Monotonie fast immer gut; die dort genannten Arbeiten sind nicht gelesen. Keeney und Raiffa 1976, Raiffa 1968, Hammond/Keeney/Raiffa (*Smart Choices*) und Clemen/Reilly (Kap. 16 und 17) sind hier nur als Fundstellen bekannt, **nicht gelesen**. **Die Bedingung der multiplikativen Form im Wortlaut ist nicht gelesen**; was diese README über Form und Kopplung sagt, ist eigene Nachrechnung in diesem Modell. Nichts ist wörtlich übernommen. <!-- lint-ok: dezimalpunkt -->

## Ehrliche Grenzen

- **Modell, kein Fall.** Auszahlungen, Servicegrade und $k_i$ sind ausgedacht; die Zufallsprobleme sind keine Fälle. Die Häufigkeiten gelten für diese Generatoren (gleichverteilte Einzelnutzen, harmonische Gewichte, Zwei-Punkt-Lose, die genannten Zähleinheiten).
- **Die Bedingungen der Form werden nicht geprüft.** „Die Ziele sind nutzenunabhängig und die Produktform gilt“ ist im Kapazitätsfall und in den Zufallsproblemen Annahme, kein Befund.
- **Kopplung nur für zwei Ziele und Zwei-Punkt-Lose gemessen.** Mit mehr Zielen ist Kopplung ein Satz von Paarungen: nicht gemessen. $K$ ist für drei Ziele gerechnet (B2), die Kopplung nicht.
- **Die Erhebung der $k_i$ ist nicht gemessen** (Rauschen, Verfahrenseffekte); die $k_i$ sind gesetzt. Zur Erhebung des Einzelnutzens siehe `nutzen-erheben-demo`.
- **Die additive Näherung ist bei sicheren Ausgängen in der Voreinstellung ($n = 20$, $m = 5$) meist unauffällig:** in 4552 von 5000 Matrizen ($\Sigma k = 0{,}5$, B5) ändert die Form Platz 1 nicht; bei den vier Werten $\Sigma k \ne 1$ ändert sie ihn in 3,3 bis 9,0 % der Matrizen, bei $m = 2$ und $n = 40$ in bis zu 882 von 3000 (29,4 %, B7). Das ist ein Negativbefund für die Wichtigkeit der Form, kein Argument dagegen: bei Risiko in den Zielen (B8) und bei Kopplung (B9) ist der Unterschied größer.
- **Kipp-Risikotoleranz nur auf einem Raster.** „Kein Wechsel“ gilt für die 17 Rasterpunkte von 5 GE bis 10⁹ GE, nicht für jedes $\rho$. $\rho^*$ ist mit dem zweiten Ziel keine Eigenschaft des Gewinns mehr, sondern hängt an $k$, Form und Kopplung.
- **Ursachen nicht untersucht:** die Richtung in $n$ (B7) und in $S$ (B8). Die Deutung „bei $c > 0$ belohnt der Kreuzterm Ausgewogenheit, bei $c < 0$ Einseitigkeit“ stammt aus den Handbeispielen (B1), keine Messung der Ursache.
- **Plattform.** Gerechnet unter Windows (Python 3.14, NumPy 2.5, SciPy 1.18) und im Linux-Test (siehe Verifikation); Zählungen als Band, nicht bitgleich.

## Verifikation

- **Kern:** Handbeispiele in Brüchen (`tests/test_algorithm.py`: $K$, Ecken, Kapazitätsfall, $q^*$, Kovarianzform, Kopplung, Sicherheitsäquivalent); Randfälle (`tests/test_edge_cases.py`: $n = 1$, $n = 2$, Gleichstand, $q$ an den Rändern, Reglergrenzen).
- **Orakel** (`tests/test_oracle_attribute_risiko.py`, anderer Rechenweg): Produktform gegen ausmultiplizierte Form (300 Bruchpaare); $K$ gegen die geschlossene Form und gegen `numpy.roots`; Erwartungsnutzen aus den Zellen gegen die Kovarianzform (Brüche) und die zu Fuß geschriebene Formel (Gleitkomma); $\rho^*$ über $u_1(30;\rho) = \tfrac12$ gegen das Sicherheitsäquivalent; linearer Einzelnutzen (Erwartungsnutzen gleich Nutzen der Erwartungswerte); die Zufallsproblem-Zählungen gegen eine Schleife über Alternativen in `fractions` (200 Probleme); Kendall-Tau gegen `scipy.stats.kendalltau`; Kopplung gegen die geschlossene Form.
- **Eigenschaften:** Umnummerieren, Spiegeln der Ziele, Ecken und Summe 1, Linearität in den Wahrscheinlichkeiten, Vorzeichen von $c$, Stetigkeit bei $\Sigma k \to 1$ (Regressionstest der Rechenfalle).
- **Zufall:** `tests/test_rng.py` (Klasse gleich Feld, Referenzwerte, Gleichverteilung); `tests/test_sweeps.py` (gleicher Zufalls-Seed gleiche Ausgabe, eigene Ströme, $\Sigma k = 1$ gibt 0).
- **Verdikte:** reine Funktionen (`tests/test_verdicts.py`, je Zweig und Zahl; `tests/test_presets.py`: je Preset die erwartete Verdikt-Zeile, das Preset $\Sigma k = 1$ meldet keinen Formunterschied).
- **Claims:** `tests/test_claims.py` rechnet jede Zahl der Tabelle B1 bis B9 neu und sucht sie in dieser README.
- **Oberfläche:** AppTest-Rauchtest (`tests/test_app.py`: jeder Button, jeder Regler am Minimum und Maximum), Layout (`tests/test_layout.py`: Schnellstart in der Seitenleiste, fester Kopf, jede Regleränderung hebt die Markierung auf, Permalink), Footer (`tests/test_footer.py`), Laufzeit-Importe (`tests/test_runtime_imports.py`).
- **Linux-Lauf:** alle 123 Tests grün in der nachgestellten Linux-CI (Docker, Python 3.12, frische Installation aus `requirements-dev.txt`, Speichergrenze 16 GB), Stand dieser README.
- **Testzahl:** 123 Tests (`pytest tests`).

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: vier Experimente, Kernfragen, Grenzen, Formeln |
| `atr_constants.py` | Regler-Grenzen, Annahmen des Kapazitätsfalls, Farben, Presets |
| `atr_evaluation.py` | Auswertung des Kapazitätsfalls, Verdikte (reine Funktionen), Formatierung |
| `atr_layout.py` | Fester Kopf (Stylesheet) |
| `atr_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufalls-Seed-Button |
| `atr_rng.py` | SplitMix64 (Klasse und vektorisiert) |
| `atr_scenario.py` | Kapazitätsfall mit zwei Zielen: Zellen, Erwartungsnutzen, Kipp-$\rho$ und Kipp-$q$ |
| `atr_sweeps.py` | Zufallsprobleme: Form, Risiko, Kopplung, Stetigkeitsprobe |
| `atr_utility.py` | Einzelnutzen, $K$, multiplikative und additive Form, Kendall-Tau |
| `atr_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): Streamlit, NumPy, SciPy, pandas, Plotly |
| `tests/` | Handbeispiele, Orakel, Eigenschaften, Claims, Verdikte, Presets, Layout, AppTest-Rauchtest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Erhebung der Skalierungskonstanten** (Lotterie mit Rauschen, Verfahrenseffekte): die $k_i$ sind gesetzt (siehe `nutzen-erheben-demo`).
- **Gewichtswechsel, Swing, SMAA, Rangumkehr und Normierung** bei sicheren Matrizen: dazu gibt es die Demos zum Mehrkriterien-Ranking, etwa [`rangumkehr-demo`](https://github.com/sebastian-hanisch/rangumkehr-demo) (sichere Matrix, keine Lotterien). Hier nur die Brücke: bei Risiko heißen die Gewichte $k_i$ und ihre Summe sagt etwas aus.
- **Kopplung für mehr als zwei Ziele**, Einflussdiagramm-Rechnung und die Frage, ob und wie sich die multiplikative Form in einem LP schreiben lässt.
- **Prüfung der Nutzenunabhängigkeit** als Bedingung; Prospect Theory und Allais/Ellsberg (siehe `erwartungsnutzen-grenze-demo`: dort geht es um die Unabhängigkeitsannahme im Erwartungsnutzen, hier um die Unabhängigkeit der Ziele; nicht verwechseln).
- **PDF-Export.**

## Nachbarn

Die anderen zehn Demos der Entscheidungsanalyse: [`baum-strategie-demo`](https://github.com/sebastian-hanisch/baum-strategie-demo), [`testwert-evsi-demo`](https://github.com/sebastian-hanisch/testwert-evsi-demo), [`kipp-schwelle-demo`](https://github.com/sebastian-hanisch/kipp-schwelle-demo), [`risiko-aequivalent-demo`](https://github.com/sebastian-hanisch/risiko-aequivalent-demo) (Einzelnutzen und $\rho^*$, hier benutzt, dort erklärt), [`information-risikoaversion-demo`](https://github.com/sebastian-hanisch/information-risikoaversion-demo), [`nutzen-erheben-demo`](https://github.com/sebastian-hanisch/nutzen-erheben-demo), [`erwartungsnutzen-grenze-demo`](https://github.com/sebastian-hanisch/erwartungsnutzen-grenze-demo), [`baum-waechst-demo`](https://github.com/sebastian-hanisch/baum-waechst-demo), [`stichprobenbaum-demo`](https://github.com/sebastian-hanisch/stichprobenbaum-demo) und [`entscheidung-verfahrenswahl-demo`](https://github.com/sebastian-hanisch/entscheidung-verfahrenswahl-demo) (zuletzt: wo Baum, Szenario-Rechnung und andere Verfahren die Form schreiben). Das Sicherheitsäquivalent im CARA-Fall rechnet auch [`kuchen-vergroessern-demo`](https://github.com/sebastian-hanisch/kuchen-vergroessern-demo) (dort als Verhandlungsnutzen, ohne Mehrziel-Form und ohne Kopplung); der Rechenweg hier ist eigen geschrieben, nichts ist importiert.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Entscheidungsanalyse: Information, Risiko, Grenzen](https://sebastianhanisch.net/konzepte-entscheidungsanalyse.html).
