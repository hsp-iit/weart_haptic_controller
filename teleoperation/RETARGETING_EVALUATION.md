# Valutazione del retargeting indice WEART -> LEAP

## Limite osservabile

Un thimble fornisce una chiusura scalare, orientamento assoluto IMU e distanza
dito-palmo ToF. Non misura separatamente MCP, PIP e DIP. Senza una misura
esterna, piu' terne articolari possono spiegare lo stesso campione: stabilita' e
coerenza non dimostrano accuratezza anatomica.

Il confronto corretto usa quindi due livelli:

1. accuratezza contro ground truth umano sincronizzato;
2. robustezza del comando quando il ground truth non e' disponibile.

## Dataset minimo

Registrare gli stessi gesti per ogni metodo, senza comandare l'hardware in
parallelo. Usare almeno 10 ripetizioni per gesto:

- apertura/chiusura lenta e veloce;
- soste di 5 s a 0%, 25%, 50%, 75% e 100%;
- chiusura parziale con forme diverse del dito;
- ritorno open dopo 10 cicli;
- rotazione del palmo a dito fermo, per misurare la contaminazione IMU;
- avvicinamento/allontanamento del palmo dal sensore, per stressare il ToF.

Registrare `/weart/index/raw`, output articolare del metodo, joint state reale
LEAP e video laterale sincronizzato. Dal video ricavare MCP/PIP/DIP umano con
marker oppure keypoint tracking. Senza questi angoli, il confronto resta una
euristica di robustezza.

## Metriche

Tutte le metriche sono "piu' basso = meglio":

- `E_pose`: RMSE medio MCP/PIP/DIP normalizzato per il rispettivo range, contro
  ground truth; e' la metrica principale.
- `J_static`: deviazione standard del comando durante ogni sosta.
- `L`: ritardo fra inizio variazione della closure umana e comando LEAP,
  stimato con correlazione incrociata.
- `D_return`: distanza dalla posa open iniziale dopo ogni ciclo.
- `P_palm`: variazione del comando durante la rotazione del palmo con dito
  fermo; isola l'errore dovuto all'IMU assoluta.
- `V_mono`: percentuale di campioni in cui la closure cresce ma la flessione
  LEAP totale diminuisce, escluse variazioni inferiori al rumore.
- `S_limits`: numero di saturazioni, salti oltre il limite di velocita' o errori
  del solver.

Aggregazione consigliata con ground truth:

```text
score = 0.50 E_pose + 0.15 J_static + 0.10 L
      + 0.10 D_return + 0.10 P_palm + 0.05 V_mono
```

Normalizzare ogni metrica con mediana e MAD calcolate sui tre metodi prima
dell'aggregazione. Scartare qualsiasi metodo con `S_limits > 0`, anche se lo
score aggregato e' basso.

Senza ground truth, rimuovere `E_pose` e rinormalizzare i pesi. Tale score dice
quale comando e' piu' stabile e coerente, non quale ricostruisce davvero le tre
articolazioni umane.

## Interpretazione attesa

- `weart_index_subscriber.py`: puo' vincere solo se lunghezze del dito, punto
  ToF e rapporto DIP/PIP sono calibrati sul soggetto.
- `weart_leap_index_ik.py`: usa correttamente il modello robot, ma IMU + ToF non
  rendono univoci tre giunti; la preferenza per DIP piccolo sceglie una soluzione
  senza provarne la correttezza anatomica.
- `weart_index_synergy.py`: problema identificabile e normalmente piu' stabile;
  non puo' riprodurre forme diverse aventi la stessa closure. E' il baseline
  consigliato finche' non e' disponibile un secondo sensore o ground truth.
