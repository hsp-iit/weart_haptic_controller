# Funzione di costo del retargeting WEART-LEAP basato su synergy

## Variabile ottimizzata

Il metodo ottimizza una sola variabile:

$$
c_k \in [0,1]
$$

dove $c_k=0$ indica l'indice aperto e $c_k=1$ l'indice completamente chiuso.

Questa scelta evita di stimare direttamente tre angoli articolari MCP, PIP e
DIP da misure che non li rendono univocamente osservabili.

## Modello di synergy

La posa della LEAP Hand viene calcolata mediante:

$$
q_i(c_k) =
\operatorname{clip}\left(
q_{i,\mathrm{closed}}\,c_k^{p_i},
q_{i,\min},
q_{i,\max}
\right)
$$

I parametri predefiniti sono:

$$
\mathbf q_{\mathrm{closed}} =
\begin{bmatrix}
85^\circ & 95^\circ & 65^\circ
\end{bmatrix}^{T}
$$

$$
\mathbf p =
\begin{bmatrix}
0.85 & 1.05 & 1.25
\end{bmatrix}^{T}
$$

Le tre componenti rappresentano rispettivamente MCP, PIP e DIP. Un esponente
minore di uno anticipa il movimento del giunto; un esponente maggiore di uno lo
ritarda.

## Residuo della closure WEART

$$
r_c(c_k) = \frac{c_k-c_k^W}{\sigma_c}
$$

dove $c_k^W$ e' la closure WEART normalizzata e:

$$
\sigma_c=0.055
$$

Questo e' il vincolo principale della fusione.

## Residuo di continuita' temporale

$$
r_s(c_k) = \frac{c_k-c_{k-1}}{\sigma_s}
$$

con:

$$
\sigma_s=0.18
$$

Il termine penalizza variazioni improvvise della closure stimata tra campioni
consecutivi.

## Residuo di orientamento IMU

$$
r_\theta(c_k) =
\sqrt{w_a}\,
\frac{
\operatorname{wrap}_{[-\pi,\pi]}\left(
\Theta(\mathbf q(c_k))-\theta_k^{\mathrm{IMU}}
\right)
}{\sigma_\theta}
$$

dove:

- $\Theta(\mathbf q)$ e' l'orientamento della falange distale ottenuto dalla
  forward kinematics del modello URDF;
- $\theta_k^{\mathrm{IMU}}$ e' l'angolo prodotto dal filtro complementare
  accelerometro-giroscopio;
- $\sigma_\theta=25^\circ$;
- $w_a$ rappresenta l'affidabilita' istantanea dell'accelerometro.

L'affidabilita' dell'accelerometro e' calcolata come:

$$
w_a =
\exp\left[
-\frac{1}{2}
\left(
\frac{\lVert\mathbf a_k\rVert-1}{0.12}
\right)^2
\right]
$$

Quando $\lVert\mathbf a_k\rVert \simeq 1g$, si ottiene $w_a \simeq 1$.
Durante forti accelerazioni, $w_a$ diminuisce e l'IMU influenza meno la stima.

## Residuo Time-of-Flight

Quando la relazione ToF-closure e' disponibile, la distanza prevista e':

$$
\hat d_k(c_k) = d_{\mathrm{open}} + \Delta d_{\mathrm{closed}}c_k
$$

Il residuo corrispondente e':

$$
r_d(c_k) =
\frac{\hat d_k(c_k)-d_k^{\mathrm{ToF}}}{\sigma_d}
$$

con:

$$
\sigma_d=12\ \mathrm{mm}
$$

La variazione ToF a chiusura completa puo' essere configurata manualmente
oppure stimata online mediante regressione pesata:

$$
\Delta d_{\mathrm{closed}} =
\frac{
\sum_j \lambda^{k-j}c_j(d_j-d_{\mathrm{open}})
}{
\sum_j \lambda^{k-j}c_j^2
}
$$

con forgetting factor:

$$
\lambda=0.995
$$

Se il campione ToF non e' valido o la regressione non ha ancora dati
sufficienti, $r_d$ viene escluso dalla funzione di costo.

## Funzione di costo quadratica equivalente

Senza loss robusta, il problema sarebbe:

$$
c_k^* =
\underset{0\leq c_k\leq1}{\operatorname{argmin}}
\left[
r_c(c_k)^2 +
r_s(c_k)^2 +
r_\theta(c_k)^2 +
\mathbb I_{\mathrm{ToF}}r_d(c_k)^2
\right]
$$

dove $\mathbb I_{\mathrm{ToF}}$ vale uno quando il ToF e' utilizzabile e zero
altrimenti.

## Funzione di costo realmente utilizzata

L'implementazione usa la loss robusta `soft_l1`. Definendo:

$$
\rho(r)=\sqrt{1+r^2}-1
$$

il problema effettivamente risolto e':

$$
\boxed{
c_k^* =
\underset{0\leq c_k\leq1}{\operatorname{argmin}}
\sum_{r_i\in\mathcal R_k}
\left(\sqrt{1+r_i(c_k)^2}-1\right)
}
$$

con:

$$
\mathcal R_k = \{r_c,r_s,r_\theta\}
$$

oppure, quando il ToF e' disponibile:

$$
\mathcal R_k = \{r_c,r_s,r_\theta,r_d\}
$$

La `soft_l1` e' quasi quadratica per residui piccoli e quasi lineare per
residui grandi. Un'anomalia IMU o ToF ha quindi un'influenza limitata rispetto
a una normale somma di errori quadratici.

## Risultato del retargeting

Dopo avere trovato $c_k^*$, il target articolare e':

$$
\mathbf q_k^* = \mathbf q(c_k^*)
$$

Il target viene infine limitato usando i limiti del modello URDF, filtrato e
sottoposto a un limite di velocita' prima dell'invio ai motori LEAP.
